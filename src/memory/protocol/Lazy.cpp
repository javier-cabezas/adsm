#include "Lazy.h"

#include <os/Memory.h>

namespace gmac { namespace memory { namespace protocol {

gmacError_t Lazy::acquire(Object &obj)
{
    SharedObject<State> &object = dynamic_cast<SharedObject<State> &>(obj);
    SharedObject<State>::SystemMap &map = object.blocks();
    SharedObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        switch(block->state()) {
            case Dirty:
            case ReadOnly:
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    return gmacErrorInvalidValue;
                block->state(ReadOnly);
                break;
            case Invalid: break;
        }
    }
    return gmacSuccess;
}

gmacError_t Lazy::release(Object &obj)
{
    SharedObject<State> &object = dynamic_cast<SharedObject<State> &>(obj);
    SharedObject<State>::SystemMap &map = object.blocks();
    SharedObject<State>::SystemMap::iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        SystemBlock<State> *block = i->second;
        switch(block->state()) {
            case Dirty:
                object.owner()->copyToDevice(object.device(block->addr()), block->addr(), block->size());
                if(Memory::protect(block->addr(), block->size(), PROT_NONE) < 0)
                    return gmacErrorInvalidValue;
                block->state(Invalid);
            break;

        case Invalid:
        case ReadOnly:
            break;
        }
    }
    return gmacSuccess;
}

gmacError_t Lazy::read(Object &obj, void *addr)
{
    SharedObject<State> &object = dynamic_cast<SharedObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    if(Memory::protect(block->addr(), block->size(), PROT_WRITE) < 0)
        return gmacErrorInvalidValue;
    object.owner()->copyToHost(block->addr(), object.device(block->addr()), block->size());
    if(Memory::protect(block->addr(), block->size(), PROT_READ) < 0)
        return gmacErrorInvalidValue;
    block->state(ReadOnly);
    return gmacSuccess;
}

gmacError_t Lazy::write(Object &obj, void *addr)
{
    SharedObject<State> &object = dynamic_cast<SharedObject<State> &>(obj);
    SystemBlock<State> *block = object.findBlock(addr);
    if(block == NULL) return gmacErrorInvalidValue;
    if(Memory::protect(block->addr(), block->size(), PROT_READ | PROT_WRITE) < 0)
        return gmacErrorInvalidValue;
    block->state(Dirty);
    return gmacSuccess;
}


} } }
