#include "core/Mode.h"
#include "memory/HostMappedObject.h"
#include "util/Logger.h"

namespace __impl { namespace memory {



bool HostMappedSet::insert(HostMappedObject *object)
{
    if(object == NULL) return false;
    uint8_t *key = (uint8_t *)object->addr() + object->size();
    lockWrite();
    std::pair<Parent::iterator, bool> ret =
        Parent::insert(Parent::value_type(key, object));
    unlock();
    return ret.second;
}

HostMappedObject *HostMappedSet::get(hostptr_t addr) const
{
    HostMappedObject *object = NULL;
    lockRead();
    Parent::const_iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr >= i->second->addr());
    if(ret) {
        object = i->second;
        object->incRef();
    }
    unlock();
    return object;
}

bool HostMappedSet::remove(hostptr_t addr)
{
    lockWrite();
    Parent::iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr == i->second->addr());
    if(ret == true) erase(i);
    unlock();
    return ret;
}

HostMappedSet HostMappedObject::set_;

HostMappedObject::HostMappedObject(core::Mode &mode, size_t size) :
    size_(size),
    mode_(mode)
{
        // Allocate memory (if necessary)
    addr_ = alloc(mode_);
    if(addr_ == NULL) return;
    set_.insert(this);
    TRACE(LOCAL, "Creating Host Mapped Object @ %p) ", addr_);
}


HostMappedObject::~HostMappedObject()
{
    if(addr_ != NULL) free(mode_);
    TRACE(LOCAL, "Destroying Host Mapped Object @ %p", addr_);
}


accptr_t HostMappedObject::acceleratorAddr(const hostptr_t addr) const
{
    accptr_t ret = accptr_t(0);
    if(addr_ != NULL) {
        unsigned offset = unsigned(addr - addr_);
        accptr_t acceleratorAddr = getAccPtr(mode_);
        ret = acceleratorAddr + offset;
    }
    return ret;
}

hostptr_t
HostMappedObject::alloc(core::Mode &mode)
{
    hostptr_t ret = NULL;
    if(mode.hostAlloc(ret, size_) != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free(core::Mode &mode)
{
    mode.hostFree(addr_);
}

accptr_t
HostMappedObject::getAccPtr(core::Mode &mode) const
{
    return mode.hostMapAddr(addr_);
}


}}
