#include "HostMappedObject.h"

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

HostMappedObject *HostMappedSet::get(void *addr) const
{
    HostMappedObject *object = NULL;
    lockRead();
    Parent::const_iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr >= i->second->addr());
    if(ret) {
        object = i->second;
        object->use();
    }
    unlock();
    return object;
}

bool HostMappedSet::remove(void *addr)
{
    lockWrite();
    Parent::iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr == i->second->addr());
    if(ret == true) erase(i);
    unlock();
    return ret;
}

HostMappedSet HostMappedObject::set_;

HostMappedObject::HostMappedObject(size_t size) :
    size_(size)
{
	// Allocate memory (if necessary)
    addr_ = (uint8_t *)HostMappedAlloc(size);
    if(addr_ == NULL) return; 
    set_.insert(this);
    TRACE(LOCAL, "Creating Host Mapped Object @ %p) ", addr_);
}


HostMappedObject::~HostMappedObject()
{
    if(addr_ != NULL) HostMappedFree(addr_);
    TRACE(LOCAL, "Destroying Host Mapped Object @ %p", addr_);
}


void *HostMappedObject::acceleratorAddr(const void *addr) const
{
    void *ret = NULL;
    if(addr_ != NULL) {
        unsigned offset = unsigned((uint8_t *)addr - addr_);
        uint8_t *acceleratorAddr = (uint8_t *)HostMappedPtr(addr_);
        ret = acceleratorAddr + offset;
    }
    return ret;
}

}}

