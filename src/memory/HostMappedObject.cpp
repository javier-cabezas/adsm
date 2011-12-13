#include "core/address_space.h"
#include "memory/memory.h"
#include "memory/HostMappedObject.h"
#include "trace/logger.h"

namespace __impl { namespace memory {

bool set_object_host_mapped::insert(object_host_mapped *object)
{
    if(object == NULL) return false;
    uint8_t *key = (uint8_t *)object->addr() + object->size();
    lock_write();
    std::pair<Parent::iterator, bool> ret =
        Parent::insert(Parent::value_type(key, object));
    unlock();
    return ret.second;
}

object_host_mapped *set_object_host_mapped::get(hostptr_t addr) const
{
    object_host_mapped *object = NULL;
    lock_read();
    Parent::const_iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr >= i->second->addr());
    if(ret) {
        object = i->second;
        object->incRef();
    }
    unlock();
    return object;
}

bool set_object_host_mapped::remove(hostptr_t addr)
{
    lock_write();
    Parent::iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr == i->second->addr());
    if(ret == true) erase(i);
    unlock();
    return ret;
}

set_object_host_mapped object_host_mapped::set_;

object_host_mapped::object_host_mapped(core::address_space_ptr aspace, size_t size) :
    util::Reference("HostMappedObject"),
    size_(size),
    owner_(aspace)
{
    gmacError_t err;
    // Allocate memory (if necessary)
    addr_ = alloc(owner_, err);
    if(err != gmacSuccess) return;
    set_.insert(this);
    TRACE(LOCAL, "Creating Host Mapped Object @ %p) ", addr_.get_host_addr());
}


object_host_mapped::~object_host_mapped()
{
    if(addr_) free(owner_);
    TRACE(LOCAL, "Destroying Host Mapped Object @ %p", addr_.get_host_addr());
}

accptr_t
object_host_mapped::get_device_addr(core::address_space_ptr current, const hostptr_t addr) const
{
    //ASSERTION(current == owner_);
    accptr_t ret = accptr_t(0);
    if(addr_) {
        unsigned offset = unsigned(addr - addr_.get_host_addr());
        accptr_t acceleratorAddr = getAccPtr(current);
        ret = acceleratorAddr + offset;
    }
    return ret;
}

hal::ptr_t
object_host_mapped::alloc(core::address_space_ptr aspace, gmacError_t &err)
{
    hal::ptr_t ret;
    //ret = Memory::map(NULL, size_, GMAC_PROT_READWRITE);
    ret = aspace->alloc_host_pinned(size_, err);
    return ret;
}

void
object_host_mapped::free(core::address_space_ptr aspace)
{
    //Memory::unmap(addr_, size_);
    aspace->free_host_pinned(addr_);
}

accptr_t
object_host_mapped::getAccPtr(core::address_space_ptr aspace) const
{
#if 0
    gmacError_t err;
    return aspace->get_host_pinned_mapping(addr_.get_host_addr(), err);
#endif
    return accptr_t(0);
}

}}
