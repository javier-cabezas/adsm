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
        switch(block->state()) {
            case Dirty:
            case ReadOnly:
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    fatal("Unable to set memory permissions");
                block->state(ReadOnly);
                break;
            case Invalid: break;
        }
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
        block->state(Invalid);
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
    }
    return ret;
}


gmacError_t Lazy::read(Object &obj, void *addr)
{
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    if(Memory::protect(block->addr(), block->size(), PROT_WRITE) < 0)
        return gmacErrorInvalidValue;
    gmacError_t ret = object.acquire(block);
    if(ret != gmacSuccess) return ret;
    if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
        return gmacErrorInvalidValue;
    block->state(ReadOnly);
    return gmacSuccess;
}

gmacError_t Lazy::write(Object &obj, void *addr)
{
    StateObject<State> &object = dynamic_cast<StateObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    if(Memory::protect(block->addr(), block->size(), PROT_READ | PROT_WRITE) < 0)
        return gmacErrorInvalidValue;
    block->state(Dirty);
    return gmacSuccess;
}


} } }
