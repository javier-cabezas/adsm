#include "core/address_space.h"
#include "memory/Memory.h"
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

HostMappedObject::HostMappedObject(util::smart_ptr<core::address_space>::shared aspace, size_t size) :
    util::Reference("HostMappedObject"),
    size_(size),
    owner_(aspace)
{
    // Allocate memory (if necessary)
    addr_ = alloc(owner_);
    if(addr_ == NULL) return;
    set_.insert(this);
    TRACE(LOCAL, "Creating Host Mapped Object @ %p) ", addr_);
}


HostMappedObject::~HostMappedObject()
{
    if(addr_ != NULL) free(owner_);
    TRACE(LOCAL, "Destroying Host Mapped Object @ %p", addr_);
}

accptr_t
HostMappedObject::get_device_addr(util::smart_ptr<core::address_space>::shared current, const hostptr_t addr) const
{
    //ASSERTION(current == owner_);
    accptr_t ret = accptr_t(0);
    if(addr_ != NULL) {
        unsigned offset = unsigned(addr - addr_);
        accptr_t acceleratorAddr = getAccPtr(current);
        ret = acceleratorAddr + offset;
    }
    return ret;
}

hostptr_t
HostMappedObject::alloc(util::smart_ptr<core::address_space>::shared aspace)
{
    hostptr_t ret = NULL;
    //ret = Memory::map(NULL, size_, GMAC_PROT_READWRITE);
    gmacError_t err;
    ret = aspace->alloc_host_pinned(size_, err);
    if (err != gmacSuccess) return NULL;
    return ret;
}

void
HostMappedObject::free(util::smart_ptr<core::address_space>::shared aspace)
{
    //Memory::unmap(addr_, size_);
    aspace->free_host_pinned(addr_);
}

accptr_t
HostMappedObject::getAccPtr(util::smart_ptr<core::address_space>::shared aspace) const
{
    gmacError_t err;
    return aspace->get_host_pinned_mapping(addr_, err);
}

}}
