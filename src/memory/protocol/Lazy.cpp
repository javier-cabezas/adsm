#include "Lazy.h"

#include <os/Memory.h>

namespace gmac { namespace memory { namespace protocol {

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
            case Dirty:
            case ReadOnly:
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    fatal("Unable to set memory permissions");
                block->state(ReadOnly);
                break;
            case Invalid: break;
        }
        block->unlock();
    }
    return ret;
}

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
                ret = object.release(block);
                if(ret != gmacSuccess) return ret;
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    fatal("Unable to set memory permissions");
                block->state(Invalid);
            break;

            case Invalid:
            case ReadOnly:
                break;
        }
        block->unlock();
    }
    return ret;
}

gmacError_t Lazy::invalidate(Object &obj)
{
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

gmacError_t Lazy::flush(Object &obj)
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
                ret = object.release(block);
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


gmacError_t Lazy::read(Object &obj, void *addr)
{
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    block->lock();
    if(Memory::protect(block->addr(), block->size(), PROT_WRITE) < 0) {
        block->unlock();
        return gmacErrorInvalidValue;
    }
    gmacError_t ret = object.acquire(block);
    if(ret != gmacSuccess) {block->unlock(); return ret; }
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
