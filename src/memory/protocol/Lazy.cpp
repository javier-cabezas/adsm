#include "Lazy.h"

#include "memory/StateObject.h"
#include "memory/SharedObject.h"
#include "memory/ReplicatedObject.h"
#include "memory/os/Memory.h"

namespace gmac { namespace memory { namespace protocol {

Object *Lazy::createObject(size_t size)
{
    return new SharedObject<Lazy::State>(size, ReadOnly);
}

#ifndef USE_MMAP
Object *Lazy::createReplicatedObject(size_t size)
{
    return new ReplicatedObject<Lazy::State>(size, ReadOnly);
}
#endif

gmacError_t Lazy::acquire(Object &obj)
{
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        switch(block->state()) {
            case Invalid:
            case ReadOnly:
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    fatal("Unable to set memory permissions");
                block->state(Invalid);
                break;
            case Dirty:
                fatal("Block in incogruent state in aquire");
                break;
        }
        block->unlock();
    }
    return ret;
}

#ifdef USE_VM
gmacError_t Lazy::acquireWithBitmap(Object &obj)
{
    Mode *mode = gmac::Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        if (bitmap.check(obj.device(block->addr()))) {
            if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                fatal("Unable to set memory permissions");
            block->state(Invalid);
        } else {
            if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
                fatal("Unable to set memory permissions");
            block->state(ReadOnly);
        }
        block->unlock();
    }
    return ret;
}
#endif

gmacError_t Lazy::release(Object &obj)
{
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        switch(block->state()) {
            case Dirty:
                ret = object.toDevice(block);
                if(ret != gmacSuccess) return ret;
                if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
                    fatal("Unable to set memory permissions");
                block->state(ReadOnly);
            break;

            case Invalid:
            case ReadOnly:
                break;
        }
        block->unlock();
    }
    return ret;
}

gmacError_t Lazy::toHost(Object &obj)
{
    abort();
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        block->state(Invalid);
        block->unlock();
    }
    if(Memory::protect(object.addr(), object.size(), PROT_NONE))
        fatal("Unable to set memory permissions");
    return gmacSuccess;
}

gmacError_t Lazy::toDevice(Object &obj)
{
    abort();
    gmacError_t ret = gmacSuccess;
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();
        switch(block->state()) {
            case Dirty:
                ret = object.toDevice(block);
                if(ret != gmacSuccess) { block->unlock(); return ret; }
                if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
                    fatal("Unable to set memory permissions");
                block->state(ReadOnly);
            break;

        case Invalid:
        case ReadOnly:
            break;
        }
        block->unlock();
    }
    return ret;
}

static size_t blockRemainder(void * blockAddr, size_t blockSize, void * ptr, size_t n)
{
    char * _ptr  = (char *) ptr;
    char * _addr = (char *) blockAddr;

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
Lazy::toIOBuffer(IOBuffer *buffer, Object &obj, void *addr, size_t n)
{
    cfatal(n <= buffer->size(), "Wrong buffer size");
    gmacError_t ret = gmacSuccess;

    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
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
                    count = blockRemainder(block->addr(), block->size(), addr, n);
                    memcpy((char *) buffer->addr() + off, (char *) addr + off, count);
                    break;

                case Invalid:
                    Mode * mode = obj.owner();
                    ret = mode->deviceToBuffer(buffer, (char *)addr + off, count);
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
Lazy::fromIOBuffer(IOBuffer *buffer, Object &obj, void *addr, size_t n)
{
    cfatal(n <= buffer->size(), "Wrong buffer size");
    gmacError_t ret = gmacSuccess;

    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    StateObject<State>::SystemMap &map = object.blocks();
    StateObject<State>::SystemMap::iterator i;
    off_t off = 0;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        block->lock();

        if ((addr >= block->addr() && addr < (char *) block->addr() + block->size()) ||
            (addr <  block->addr() && (char *) addr + n > block->addr())) {
            size_t count = blockRemainder(block->addr(), block->size(), addr, n);

            switch(block->state()) {
                case Dirty:
                    memcpy((char *) addr + off, (char *) buffer->addr() + off, count);
                    break;

                case ReadOnly:
                    if(Memory::protect(block->addr(), block->size(), PROT_WRITE) < 0)
                        fatal("Unable to set memory permissions");
                    memcpy((char *) addr + off, (char *) buffer->addr() + off, count);
                    if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
                        fatal("Unable to set memory permissions");
                    ret = object.toDevice(block);
                    break;

                case Invalid:
                    Mode *mode = Mode::current();
                    ret = mode->bufferToDevice((char *)addr + off, buffer, count);
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
Lazy::toPointer(void *dst, const void *src, const Object &srcObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

    return ret;
}

gmacError_t
Lazy::fromPointer(void *dst, const void *src, Object &dstObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

    return ret;
}

gmacError_t
Lazy::copy(void *dst, const void *src, Object &dstObj, const void *srcObj, size_t n)
{
    gmacError_t ret = gmacSuccess;

    return ret;
}

gmacError_t
Lazy::memset(void *s, int c, size_t n)
{
    gmacError_t ret = gmacSuccess;

    return ret;
}

gmacError_t Lazy::read(Object &obj, void *addr)
{
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    block->lock();
#ifdef USE_VM
    Mode *mode = Mode::current();
    vm::Bitmap &bitmap = mode->dirtyBitmap();
   if (bitmap.checkAndClear(obj.device(block->addr()))) {
#endif
        if(Memory::protect(block->addr(), block->size(), PROT_WRITE) < 0) {
            block->unlock();
            return gmacErrorInvalidValue;
        }
        gmacError_t ret = object.toHost(block);
        if(ret != gmacSuccess) { block->unlock(); return ret; }
#ifdef USE_VM
    }
#endif
    if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0) {
        block->unlock();
        return gmacErrorInvalidValue;
    }
    block->state(ReadOnly);
    block->unlock();
    return gmacSuccess;
}

gmacError_t Lazy::write(Object &obj, void *addr)
{
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    block->lock();
    if(Memory::protect(block->addr(), block->size(), PROT_READ | PROT_WRITE) < 0) {
        block->unlock();
        return gmacErrorInvalidValue;
    }
    block->state(Dirty);
    block->unlock();
    return gmacSuccess;
}

#ifndef USE_MMAP
bool Lazy::requireUpdate(Block *b)
{
    bool ret = true;
    b->lock();
    SystemBlock<State> *block = dynamic_cast<SystemBlock<State> *>(b);
    switch(block->state()) {
        case Dirty:
            ret = false; break;
        case Invalid:
        case ReadOnly:
            ret = true; break;
    }
    b->unlock();
    return ret;
}

#endif


} } }
