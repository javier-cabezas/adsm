#include "Lazy.h"

#include "core/IOBuffer.h"

#include "memory/StateObject.h"
#include "memory/SharedObject.h"
#include "memory/ReplicatedObject.h"
#include "memory/os/Memory.h"

#include "trace/Function.h"

namespace gmac { namespace memory { namespace protocol {

List Lazy::GlobalCache_;

bool List::empty() const
{
    bool ret = std::list<Entry>::empty();
    return ret;
}

size_t List::size() const
{
    size_t ret = std::list<Entry>::size();
    return ret;
}

void List::purge(const StateObject<Lazy::State> &obj)
{
    iterator i;
    for(i = begin(); i != end(); ) {
        if(&(i->object) == &obj) i = erase(i);
        else i++;
    }
}

void List::push(const StateObject<Lazy::State> &obj,
        SystemBlock<Lazy::State> *block)
{
    push_back(Entry(obj, block));
}

Entry List::pop() 
{
    Entry ret = front();
    pop_front();
    return ret;
}

Lazy::Lazy(unsigned limit)
    : RWLock("Lazy"), _maxListSize(limit)
{
}

Lazy::~Lazy()
{
}

Object *Lazy::createObject(size_t size)
{
    return new SharedObject<Lazy::State>(size, ReadOnly);
}

void Lazy::deleteObject(const Object &obj)
{
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    lockRead();
    iterator i;
    for(i = begin(); i != end(); i++) i->second.purge(object);
    unlock();
}


#ifndef USE_MMAP
Object *Lazy::createReplicatedObject(size_t size)
{
    return new ReplicatedObject<Lazy::State>(size, ReadOnly);
}
#endif

gmacError_t Lazy::acquire(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        switch(block->state()) {
            case Invalid:
            case ReadOnly:
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    Fatal("Unable to set memory permissions");
                block->state(Invalid);
                break;
            case Dirty:
                Fatal("Block in incongruent state in acquire: %p", block->addr());
                break;
        }
        block->unlock();
    }
    return ret;
}

#ifdef USE_VM
gmacError_t Lazy::acquireWithBitmap(const Object &obj)
{
    Mode &mode = gmac::Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        if (bitmap.check(obj.device(block->addr()))) {
            if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                Fatal("Unable to set memory permissions");
            block->state(Invalid);
        } else {
            if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
                Fatal("Unable to set memory permissions");
            block->state(ReadOnly);
        }
        block->unlock();
    }
    return ret;
}
#endif

gmacError_t Lazy::release(const StateObject<State> &object, SystemBlock<State> &block)
{
    trace("Releasing block %p", block.addr());
    gmacError_t ret = gmacSuccess;
    block.lock();
    switch(block.state()) {
        case Dirty:
            ret = object.toDevice(block);
            if(ret != gmacSuccess) return ret;
            if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
            block.state(ReadOnly);
            break;

        case Invalid:
        case ReadOnly:
            break;
    }
    block.unlock();
    return ret;
}


gmacError_t Lazy::release()
{
    // Release global dirty blocks
    GlobalCache_.lockWrite();
    while(GlobalCache_.empty() == false) {
        Entry e = GlobalCache_.pop();
        gmacError_t ret = release(e.object, *e.block);
        if(ret != gmacSuccess) {
            GlobalCache_.unlock();
            return ret;
        }
    }
    GlobalCache_.unlock();

    // Get the dirty list for the current mode
    lockRead();
    iterator i = find(&gmac::Mode::current());
    if(i == end()) {
        unlock();
        return gmacSuccess;
    }
    List &list = i->second;
    unlock();

    list.lockWrite();
    // Release dirty blocks
    while(list.empty() == false) {
        Entry e = list.pop();
        gmacError_t ret = release(e.object, *e.block);
        if(ret != gmacSuccess) {
            list.unlock();
            return ret;
        }
    }
    list.unlock();

    return gmacSuccess;
}

gmacError_t Lazy::toHost(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();
        switch(block.state()) {
            case Invalid:
                if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toHost(block);
                if(ret != gmacSuccess) { block.unlock(); return ret; }
                block.state(Dirty);
                ret = addDirty(object, block, false);
                break;

            case Dirty:
                break;
            case ReadOnly:
                ret = addDirty(object, block, false);
                break;
        }
        block.unlock();
    }
    return ret;
}

gmacError_t Lazy::toDevice(const Object &obj)
{
    gmacError_t ret = gmacSuccess;
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();
        switch(block.state()) {
            case Dirty:
                ret = object.toDevice(block);
                if(ret != gmacSuccess) { block.unlock(); return ret; }
                if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                block.state(ReadOnly);
            break;

        case Invalid:
        case ReadOnly:
            break;
        }
        block.unlock();
    }
    return ret;
}

static size_t blockRemainder(const void * blockAddr, size_t blockSize, const void * ptr, size_t n)
{
    const char * _ptr  = (const char *) ptr;
    const char * _addr = (const char *) blockAddr;

    if (_ptr >= _addr && _ptr + n <  _addr + blockSize) {
        return n;
    } else if (_ptr <  _addr && _ptr + n < _addr + blockSize) {
        return _ptr + n - _addr;
    } else if (_ptr <  _addr && _ptr + n >= _addr + blockSize) {
        return blockSize;
    } else { // if (_ptr >= _addr && _ptr + n >= _addr + blockSize) {
        return _addr + blockSize - _ptr;
    }
}

gmacError_t
Lazy::toIOBuffer(IOBuffer &buffer, const Object &obj, const void *addr, size_t n)
{
    CFatal(n <= buffer.size(), "Wrong buffer size");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();

        if ((addr >= block->addr() && addr < (char *) block->addr() + block->size()) ||
            (addr <  block->addr() && (char *) addr + n > block->addr())) {
            size_t count = blockRemainder(block->addr(), block->size(), addr, n);

            switch(block->state()) {
                case Dirty:
                case ReadOnly:
                    memcpy((char *) buffer.addr() + off, (char *) addr + off, count);
                    break;

                case Invalid:
                    Process &proc = Process::getInstance();
                    Mode &mode = obj.owner();
                    ret = mode.deviceToBuffer(buffer, proc.translate((char *)addr + off), count, off);
                    if(ret != gmacSuccess) { block->unlock(); return ret; }

                    break;
            }
            off += count;
        }

        block->unlock();
    }

    return ret;
}

gmacError_t
Lazy::fromIOBuffer(IOBuffer &buffer, const Object &obj, void *addr, size_t n)
{
    CFatal(n <= buffer.size(), "Wrong buffer size");
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();

        if ((addr >= block.addr() && addr < (char *) block.addr() + block.size()) ||
            (addr <  block.addr() && (char *) addr + n > block.addr())) {
            size_t count = blockRemainder(block.addr(), block.size(), addr, n);

            switch(block.state()) {
                case Dirty:
                    memcpy((char *) addr + off, (char *) buffer.addr() + off, count);
                    break;

                case ReadOnly:
                    if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                        Fatal("Unable to set memory permissions");
                    memcpy((char *) addr + off, (char *) buffer.addr() + off, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                        Fatal("Unable to set memory permissions");
                    ret = object.toDevice(block);
                    break;

                case Invalid:
                    Mode &mode = Mode::current();
                    ret = mode.bufferToDevice((char *)addr + off, buffer, count, off);
                    if(ret != gmacSuccess) { block.unlock(); return ret; }
                    break;
            }           
            off += count;
        }

        block.unlock();
    }

    return ret;
}

gmacError_t
Lazy::toPointer(void *dst, const void *src, const Object &srcObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(srcObj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();

        if ((src >= block->addr() && src < (char *) block->addr() + block->size()) ||
            (src <  block->addr() && (char *) src + n > block->addr())) {
            size_t count = blockRemainder(block->addr(), block->size(), src, n);

            switch(block->state()) {
                case Dirty:
                case ReadOnly:
                    memcpy((char *)dst + off, (char *)src + off, count);
                    break;

                case Invalid:
                    Mode & mode = srcObj.owner();
                    ret = mode.copyToHost(dst, (char *)src + off, count);
                    if(ret != gmacSuccess) { block->unlock(); return ret; }

                    break;
            }
            off += count;
        }

        block->unlock();
    }

    return ret;
}

gmacError_t
Lazy::fromPointer(void *dst, const void *src, const Object &dstObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(dstObj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();

        if ((dst >= block.addr() && dst < (char *)block.addr() + block.size()) ||
            (dst <  block.addr() && (char *)dst + n > block.addr())) {
            size_t count = blockRemainder(block.addr(), block.size(), dst, n);

            switch(block.state()) {
                case Dirty:
                    memcpy((char *) dst + off, (char *) src + off, count);
                    break;

                case ReadOnly:
                    if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                        Fatal("Unable to set memory permissions");
                    memcpy((char *)dst + off, (char *)src + off, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                        Fatal("Unable to set memory permissions");
                    ret = object.toDevice(block);
                    break;

                case Invalid:
                    Mode &mode = Mode::current();
                    ret = mode.copyToDevice((char *)dst + off, src, count);
                    if(ret != gmacSuccess) { block.unlock(); return ret; }
                    break;
            }           
            off += count;
        }

        block.unlock();
    }

    return ret;
} 

gmacError_t
Lazy::copy(void *dst, const void *src, const Object &dstObj, const Object &srcObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

#if 0
    const StateObject<State> &dstObject = dynamic_cast<const StateObject<State> &>(dstObj);
    const StateObject<State> &srcObject = dynamic_cast<const StateObject<State> &>(srcObj);

    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    Mode &mode = Mode::current();

    gmac::IOBuffer *toHostBuffer = NULL;
    gmac::IOBuffer *toDeviceBuffer = NULL;

    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();

        if ((src >= block.addr() && src < (char *) block.addr() + block.size()) ||
            (src <  block.addr() && (char *) src + n > block.addr())) {
            size_t count = blockRemainder(block.addr(), block.size(), src, n);

            switch(block.state()) {
                case Dirty:
                case ReadOnly:
                    if (toDeviceBuffer == NULL) toDeviceBuffer = proc.createIOBuffer(count);
                    ::memcpy(buffer->addr() + off, src + off, count);
                    break;

                    ret = mode.memset((char *)s + off, c, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                        Fatal("Unable to set memory permissions");
                    ::memset((char *) s + off, c, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                        Fatal("Unable to set memory permissions");
                    ret = object.toDevice(block);
                    break;

                case Invalid:
                    ret = mode.memset((char *)s + off, c, count);
                    if(ret != gmacSuccess) { block.unlock(); return ret; }
                    break;
            }           
            off += count;
        }

        block.unlock();
    }

#endif
    return ret;
}

gmacError_t
Lazy::memset(const Object &obj, void *s, int c, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    off_t off = 0;
    Mode &mode = Mode::current();
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> &block = *i->second;
        block.lock();

        if ((s >= block.addr() && s < (char *) block.addr() + block.size()) ||
            (s <  block.addr() && (char *) s + n > block.addr())) {
            size_t count = blockRemainder(block.addr(), block.size(), s, n);

            switch(block.state()) {
                case Dirty:
                    ::memset((char *) s + off, c, count);
                    break;

                case ReadOnly:
                    ret = mode.memset((char *)s + off, c, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                        Fatal("Unable to set memory permissions");
                    ::memset((char *) s + off, c, count);
                    if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                        Fatal("Unable to set memory permissions");
                    ret = object.toDevice(block);
                    break;

                case Invalid:
                    ret = mode.memset((char *)s + off, c, count);
                    if(ret != gmacSuccess) { block.unlock(); return ret; }
                    break;
            }           
            off += count;
        }

        block.unlock();
    }

    return ret;
}

gmacError_t
Lazy::moveTo(Object &obj, Mode &mode)
{
    gmacError_t ret = gmacSuccess;
    ret = toHost(obj);
    if (ret != gmacSuccess) return ret;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    ret = object.realloc(mode);
    return ret;
}

gmacError_t Lazy::read(const Object &obj, void *addr)
{
    trace::Function::start("Lazy", "read");
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) {
        trace::Function::end("Lazy");
        return gmacErrorInvalidValue;
    }
    block->lock();

    if (block->state() != Invalid) {
        // Somebody already fixed it
        block->unlock();
        trace::Function::end("Lazy");
        return gmacSuccess;
    }
    void * tmp;
#ifdef USE_VM
    Mode &mode = Mode::current();
    vm::Bitmap &bitmap = mode.dirtyBitmap();
    if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
        tmp = Memory::map(NULL, block->size(), PROT_WRITE);
        if (tmp == NULL) {
            block->unlock();
            trace::Function::end("Lazy");
            return gmacErrorInvalidValue;
        }

        gmacError_t ret = object.toHost(*block, tmp);
        if(ret != gmacSuccess) {
            Memory::unmap(tmp, block->size());
            block->unlock();
            return ret;
        }
            
#ifdef USE_VM
    }
#endif
    Memory::protect(tmp, block->size(), PROT_READ);
    void *old = Memory::remap(tmp, block->addr(), block->size());
    if (old != block->addr()) {
        block->unlock();
        trace::Function::end("Lazy");
        return gmacErrorInvalidValue;
    }
    block->state(ReadOnly);
    block->unlock();
    trace::Function::end("Lazy");
    return gmacSuccess;
}

gmacError_t Lazy::write(const Object &obj, void *addr)
{
    trace::Function::start("Lazy", "write");
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    void *old;
    gmacError_t ret = gmacSuccess;
    if(block == NULL) {
        trace::Function::end("Lazy");
        return gmacErrorInvalidValue;
    }
    block->lock();
    Mode &mode = Mode::current();
    switch (block->state()) {
        case Dirty:
            // Somebody already fixed it
            block->unlock();
            trace::Function::end("Lazy");
            return gmacSuccess;
        case Invalid:
            void * tmp;
#ifdef USE_VM
            vm::Bitmap &bitmap = mode.dirtyBitmap();
            if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
                tmp = Memory::map(NULL, block->size(), PROT_READ | PROT_WRITE);
                if (tmp == NULL) {
                    block->unlock();
                    trace::Function::end("Lazy");
                    return gmacErrorInvalidValue;
                }

                ret = object.toHost(*block, tmp);
                if(ret != gmacSuccess) {
                    Memory::unmap(tmp, block->size());
                    block->unlock();
                    trace::Function::end("Lazy");
                    return ret;
                }
#ifdef USE_VM
            }
#endif
            old = Memory::remap(tmp, block->addr(), block->size());
            if (old != block->addr()) {
                block->unlock();
                trace::Function::end("Lazy");
                return gmacErrorInvalidValue;
            }
            break;
        case ReadOnly:
            Memory::protect(block->addr(), block->size(), PROT_READ | PROT_WRITE);
            break;
    }
    block->state(Dirty);
    trace("Setting block %p to dirty state", block->addr());
    block->unlock();
    ret = addDirty(object, *block);
    trace::Function::end("Lazy");
    return gmacSuccess;
}

gmacError_t
Lazy::addDirty(const StateObject<State> &object, SystemBlock<State> &block, bool checkOverflow)
{
    Mode &mode = object.owner();

    if (object.local()) {
        lockRead();
        iterator i = find(&mode);
        if(i == end())
            i = insert(value_type(&mode, List())).first;
        assertion(i != end());
        List &list = i->second;
        unlock();

        list.lockWrite();
        list.push(object, &block);

        if (checkOverflow) {
            // Release dirty blocks
            while(list.size() > _maxListSize) {
                Entry e = list.pop();
                gmacError_t ret = release(e.object, *e.block);
                if(ret != gmacSuccess) {
                    list.unlock();
                    return ret;
                }
            }
        }
        list.unlock();
    } else {
        GlobalCache_.lockWrite();
        GlobalCache_.push(object, &block);

        if (checkOverflow) {
            // Release dirty blocks
            while(GlobalCache_.size() > _maxListSize) {
                Entry e = GlobalCache_.pop();
                gmacError_t ret = release(e.object, *e.block);
                if(ret != gmacSuccess) {
                    GlobalCache_.unlock();
                    return ret;
                }
            }
        }
        GlobalCache_.unlock();
    }
    return gmacSuccess;
}

#ifndef USE_MMAP
bool Lazy::requireUpdate(Block &b)
{
    bool ret = true;
    b.lock();
    SystemBlock<State> &block = dynamic_cast<SystemBlock<State> &>(b);
    switch(block.state()) {
        case Dirty:
            ret = false; break;
        case Invalid:
        case ReadOnly:
            ret = true; break;
    }
    b.unlock();
    return ret;
}

#endif

}}}
