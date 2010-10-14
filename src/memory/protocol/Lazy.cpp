#include <algorithm>

#include "Lazy.h"

#include "core/IOBuffer.h"

#include "config/config.h"

#include "memory/StateObject.h"
#include "memory/SharedObject.h"
#include "memory/ReplicatedObject.h"

#if defined(POSIX)
#include "memory/posix/Memory.h"
#elif defined(WINDOWS)
#include "memory/windows/Memory.h"
#endif

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
    lockWrite();
    for(iterator i = begin(); i != end(); i++)
        delete i->second;
    unlock();
}

Object *Lazy::createObject(size_t size)
{
    Object *ret = new SharedObject<Lazy::State>(size, ReadOnly);
    ret->init();
    return ret;
}

void Lazy::deleteObject(const Object &obj)
{
    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    lockRead();
    iterator i;
    for(i = begin(); i != end(); i++) i->second->purge(object);
    unlock();
}


#ifndef USE_MMAP
Object *Lazy::createReplicatedObject(size_t size)
{
    Object *ret = new ReplicatedObject<Lazy::State>(size, ReadOnly);
    ret->init();
    return ret;
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
            ret = object.toAccelerator(block);
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
    List &list = *i->second;
    unlock();

    list.lockWrite();
    trace("Cache contains %zd elements\n", list.size());
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
                ret = object.toAccelerator(block);
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

static size_t blockRemainder(const uint8_t * blockAddr, size_t blockSize, const uint8_t * ptr, size_t n)
{
    if (ptr >= blockAddr && ptr + n <  blockAddr + blockSize) {
        return n;
    } else if (ptr <  blockAddr && ptr + n < blockAddr + blockSize) {
        return ptr + n - blockAddr;
    } else if (ptr <  blockAddr && ptr + n >= blockAddr + blockSize) {
        return blockSize;
    } else { // if (ptr >= blockAddr && ptr + n >= blockAddr + blockSize) {
        return blockAddr + blockSize - ptr;
    }
}

gmacError_t
Lazy::toIOBuffer(IOBuffer &buffer, unsigned bufferOff, const Object &obj, unsigned objectOff, size_t n)
{
    assertion(objectOff < obj.size());
    assertion(bufferOff < buffer.size());
    assertion(n <= buffer.size());
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * addr = object.addr() + objectOff;
    i = object.getBlockIterator(addr);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t count = blockRemainder(block.addr(), block.size(), addr, n);
        unsigned blockOff = addr + off - block.addr();

        switch(block.state()) {
            case Dirty:
            case ReadOnly:
                ::memcpy(buffer.addr() + bufferOff + off, addr + off, count);
                break;

            case Invalid:
                ret = object.toHostBuffer(block, blockOff, buffer, bufferOff + off, count);
                if(ret != gmacSuccess) {
                    block.unlock();
                    return ret;
                }
                break;
        }
        off += count;
        block.unlock();
        i++;
    } while (off < n);

    return ret;
}

gmacError_t
Lazy::fromIOBuffer(const Object &obj, unsigned objectOff, IOBuffer &buffer, unsigned bufferOff, size_t n)
{
    assertion(objectOff < obj.size());
    assertion(bufferOff < buffer.size());
    assertion(n <= buffer.size());
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * addr = object.addr() + objectOff;
    i = object.getBlockIterator(addr);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t count = blockRemainder(block.addr(), block.size(), addr, n);
        unsigned blockOff = addr + off - block.addr();

        switch(block.state()) {
            case Dirty:
                ::memcpy(addr + off, buffer.addr() + bufferOff + off, count);
                break;

            case ReadOnly:
                if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ::memcpy(addr + off,  buffer.addr() + off, count);
                if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = object.toAcceleratorFromBuffer(block, blockOff, buffer, bufferOff + off, count);
                if(ret != gmacSuccess) {
                    block.unlock();
                    return ret;
                }
                break;
        }           
        off += count;
        block.unlock();
        i++;
    } while (off < n);

    return ret;
}

gmacError_t
Lazy::toPointer(void *_dst, const Object &objSrc, unsigned objectOff, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(objSrc);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * dst = (uint8_t *)_dst;
    const uint8_t * src = (const uint8_t *) object.addr() + objectOff;
    i = object.getBlockIterator(src);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t count = blockRemainder(block.addr(), block.size(), src, n);
        unsigned blockOff = src + off - block.addr();

        switch(block.state()) {
            case Dirty:
            case ReadOnly:
                ::memcpy(dst + off, src + off, count);
                break;

            case Invalid:
                ret = object.toHostPointer(block, blockOff, dst + off, count);
                if(ret != gmacSuccess) { block.unlock(); return ret; }

                break;
        }
        off += count;
        block.unlock();
        i++;
    } while (off < n);

    return ret;
}

gmacError_t
Lazy::fromPointer(const Object &objDst, unsigned objectOff, const void *_src, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(objDst);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * dst = object.addr() + objectOff;
    const uint8_t * src = (const uint8_t *) _src;
    i = object.getBlockIterator(dst);
    assertion(i != map.end());

    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t count = blockRemainder(block.addr(), block.size(), dst, n);
        unsigned blockOff = dst + off - block.addr();

        switch(block.state()) {
            case Dirty:
                ::memcpy(dst + off, src + off, count);
                break;

            case ReadOnly:
                if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ::memcpy(dst + off, src + off, count);
                if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = object.toAcceleratorFromPointer(block, blockOff, src + off, count);
                if(ret != gmacSuccess) { block.unlock(); return ret; }
                break;
        }           
        off += count;
        block.unlock();
        i++;
    } while (off < n);

    return ret;
} 

inline
gmacError_t
Lazy::copyHostToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                      const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to host memory
    ::memcpy(blockDst.addr() + blockOffDst,
             blockSrc.addr() + blockOffSrc, count);
    return gmacSuccess;
}

gmacError_t
Lazy::copyHostToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                         const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to host memory AND accelerator memory
    uint8_t *tmp = (uint8_t *)Memory::map(NULL, blockDst.size(), PROT_READ | PROT_WRITE);
    if (tmp == NULL) {
        return gmacErrorInvalidValue;
    }

    if (blockOffDst > 0) {
        ::memcpy(tmp, blockDst.addr(), blockOffDst);
    }
    ::memcpy(tmp + blockOffDst, blockSrc.addr() + blockOffSrc, count);
    if (blockDst.addr() + blockOffDst + count < blockDst.end() - 1) {
        ::memcpy(tmp + blockOffDst + count,
                 blockDst.addr() + blockOffDst + count,
                 blockDst.end() - (blockDst.addr() + blockOffDst + count));
    }
    if(Memory::protect(tmp, blockDst.size(), PROT_READ) < 0)
        Fatal("Unable to set memory permissions");

    uint8_t *old = (uint8_t *)Memory::remap(tmp, blockDst.addr(), blockDst.size());
    if (old != blockDst.addr()) {
        return gmacErrorInvalidValue;
    }
    return gmacSuccess;
}

inline
gmacError_t
Lazy::copyHostToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                        const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Host memory to accelerator memory
    return objectDst.toAcceleratorFromPointer(blockDst, blockOffDst, blockSrc.addr() + blockOffSrc, count);
}


gmacError_t
Lazy::copyAcceleratorToDirty(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                             const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    // Accelerator memory to host memory
    return objectSrc.toHostPointer(blockSrc, blockOffSrc, blockSrc.addr() + blockOffSrc, count);
}

gmacError_t
Lazy::copyAcceleratorToReadOnly(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                                const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    uint8_t *tmp = (uint8_t *)Memory::map(NULL, blockDst.size(), PROT_READ | PROT_WRITE);
    if (tmp == NULL) {
        return gmacErrorInvalidValue;
    }

    if (blockOffDst > 0) {
        ::memcpy(tmp, blockDst.addr(), blockOffDst);
    }
    gmacError_t ret = objectSrc.toHostPointer(blockSrc, blockOffSrc, tmp + blockOffSrc, count);
    if(ret != gmacSuccess) {
        blockDst.unlock();
    }
    if (blockDst.addr() + blockOffDst + count < blockDst.end() - 1) {
        ::memcpy(tmp + blockOffDst + count,
                 blockDst.addr() + blockOffDst + count,
                 blockDst.end() - (blockDst.addr() + blockOffDst + count));
    }
    if(Memory::protect(tmp, blockDst.size(), PROT_READ) < 0)
        Fatal("Unable to set memory permissions");

    void *old = Memory::remap(tmp, blockDst.addr(), blockDst.size());
    if (old != blockDst.addr()) {
        return gmacErrorInvalidValue;
    }

    ret = objectDst.toAccelerator(blockDst, blockOffDst, count);
    return ret;
}

gmacError_t
Lazy::copyAcceleratorToInvalid(const StateObject<State> &objectDst, Block &blockDst, unsigned blockOffDst,
                               const StateObject<State> &objectSrc, Block &blockSrc, unsigned blockOffSrc, size_t count)
{
    Process &p = Process::getInstance();
    IOBuffer *buffer = p.createIOBuffer(count); 
    if (!buffer) {
        void *tmp = Memory::map(NULL, count, PROT_READ | PROT_WRITE);
        CFatal(tmp != NULL, "Unable to set memory permissions");
       gmacError_t ret = objectSrc.toHostPointer(blockSrc, blockOffSrc, tmp, count);
       if (ret != gmacSuccess) return ret;
       ret = objectDst.toAcceleratorFromPointer(blockDst, blockOffDst, tmp, count);
       if (ret != gmacSuccess) return ret;
    } else {
       gmacError_t ret = objectSrc.toHostBuffer(blockSrc, blockOffSrc, *buffer, 0, count);
       if (ret != gmacSuccess) return ret;
       ret = buffer->wait();
       if (ret != gmacSuccess) return ret;
       ret = objectDst.toAcceleratorFromBuffer(blockDst, blockOffDst, *buffer, 0, count);
       if (ret != gmacSuccess) return ret;
       ret = buffer->wait();
       if (ret != gmacSuccess) return ret;
       p.destroyIOBuffer(buffer);
    }
    return gmacSuccess;
}

gmacError_t
Lazy::copy(const Object &objDst, unsigned offDst, const Object &objSrc, unsigned offSrc, size_t n)
{
    gmacError_t ret = gmacSuccess;
    
    const StateObject<State> &objectDst = dynamic_cast<const StateObject<State> &>(objDst);
    const StateObject<State> &objectSrc = dynamic_cast<const StateObject<State> &>(objSrc);

    const StateObject<State>::SystemMap &mapDst = objectDst.blocks();
    const StateObject<State>::SystemMap &mapSrc = objectSrc.blocks();
    unsigned off = 0;

    gmac::IOBuffer *toHostBuffer = NULL;
    gmac::IOBuffer *toDeviceBuffer = NULL;

    uint8_t * dst = objectDst.addr() + offDst;
    uint8_t * src = objectSrc.addr() + offSrc;

    StateObject<State>::SystemMap::const_iterator iDst = objectDst.getBlockIterator(dst);
    StateObject<State>::SystemMap::const_iterator iSrc = objectSrc.getBlockIterator(src);
    assertion(iDst != mapDst.end());
    assertion(iSrc != mapSrc.end());
    do {
        SystemBlock<State> &blockDst = *iDst->second;
        SystemBlock<State> &blockSrc = *iSrc->second;
        blockDst.lock();
        blockSrc.lock();

        size_t countDst = blockRemainder(blockDst.addr(), blockDst.size(), dst, n);
        size_t countSrc = blockRemainder(blockSrc.addr(), blockSrc.size(), src, n);

        size_t count = countSrc != countDst? std::min(countSrc, countDst): countSrc;

        unsigned blockOffDst = dst + off - blockDst.addr();
        unsigned blockOffSrc = src + off - blockSrc.addr();

        switch(blockSrc.state()) {
            uint8_t * tmp;
            uint8_t * old;
            // Source location present in host memory
            case Dirty:
            case ReadOnly:
                switch(blockDst.state()) {
                    case Dirty:
                        ret = copyHostToDirty(objectDst, blockDst, blockOffDst,
                                              objectSrc, blockSrc, blockOffSrc, count);
                        break;
                    case ReadOnly:
                        ret = copyHostToReadOnly(objectDst, blockDst, blockOffDst,
                                                 objectSrc, blockSrc, blockOffSrc, count);
                        break;
                    case Invalid:
                        ret = copyHostToInvalid(objectDst, blockDst, blockOffDst,
                                                objectSrc, blockSrc, blockOffSrc, count);
                        break;
                }
                break;

            // Source location in accelerator memory
            case Invalid:
                switch(blockDst.state()) {
                    case Dirty:
                        ret = copyHostToDirty(objectDst, blockDst, blockOffDst,
                                              objectSrc, blockSrc, blockOffSrc, count);
                        break;
                    // Accelerator memory to host memory AND accelerator memory
                    case ReadOnly:
                        ret = copyHostToReadOnly(objectDst, blockDst, blockOffDst,
                                                 objectSrc, blockSrc, blockOffSrc, count);
                        break;
                    // Host memory to accelerator memory
                    case Invalid:
                        ret = copyHostToInvalid(objectDst, blockDst, blockOffDst,
                                                objectSrc, blockSrc, blockOffSrc, count);
                        break;
                }
                break;
        }           
        blockDst.unlock();
        blockSrc.unlock();
        if(ret != gmacSuccess) {
            return ret;
        }
        off += count;
        if (blockDst.addr() + countDst == blockDst.end()) {
            iDst++;
        }
        if (blockSrc.addr() + countSrc == blockSrc.end()) {
            iSrc++;
        }
    } while (off < n);

    return ret;
}

gmacError_t
Lazy::memset(const Object &obj, unsigned objectOff, int c, size_t n)
{
    gmacError_t ret = gmacSuccess;

    const StateObject<State> &object = dynamic_cast<const StateObject<State> &>(obj);
    const StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::const_iterator i;
    unsigned off = 0;
    uint8_t * s = object.addr() + objectOff;
    i = object.getBlockIterator(s);
    assert(i != map.end());
    
    do {
        SystemBlock<State> &block = *i->second;
        block.lock();

        size_t count = blockRemainder(block.addr(), block.size(), s, n);

        switch(block.state()) {
            case Dirty:
                ::memset(s + off, c, count);
                break;

            case ReadOnly:
                ret = obj.owner().memset(s + off, c, count);
                if(Memory::protect(block.addr(), block.size(), PROT_WRITE) < 0)
                    Fatal("Unable to set memory permissions");
                ::memset(s + off, c, count);
                if(Memory::protect(block.addr(), block.size(), PROT_READ) < 0)
                    Fatal("Unable to set memory permissions");
                ret = object.toAccelerator(block);
                break;

            case Invalid:
                ret = obj.owner().memset(s + off, c, count);
                if(ret != gmacSuccess) {
                    block.unlock();
                    return ret;
                }
                break;
        }           
        block.unlock();
        off += count;
        i++;
    } while (off < n);

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

gmacError_t Lazy::signalRead(const Object &obj, void *addr)
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

        gmacError_t ret = object.toHostPointer(*block, 0, tmp, block->size());
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

gmacError_t Lazy::signalWrite(const Object &obj, void *addr)
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

                ret = object.toHostPointer(*block, 0, tmp, block->size());
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

    // Handle local mode allocations
    if (object.isLocal()) {
        lockWrite();
        iterator i = find(&mode);
        if(i == end()) {
            i = insert(value_type(&mode, new List())).first;
        }
        assertion(i != end());
        List &list = *i->second;
        unlock();

        list.lockWrite();
        list.push(object, &block);

        trace("Adding %p to dirty list %p (%zd)", block.addr(), &list, list.size());

        if (checkOverflow) {
            // Release dirty blocks
            while(list.size() > _maxListSize) {
                Entry e = list.pop();
                trace("Eagerly transferring %p", e.block->addr());
                gmacError_t ret = release(e.object, *e.block);
                if(ret != gmacSuccess) {
                    list.unlock();
                    return ret;
                }
            }
        }
        list.unlock();
    } else { // Handle global allocations
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
