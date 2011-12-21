#include "core/address_space.h"

#include "memory/memory.h"
#include "memory/object_mapped.h"

#include "trace/logger.h"

namespace __impl { namespace memory {

bool set_object_mapped::add_object(object_mapped &obj)
{
    uint8_t *key = (uint8_t *)obj.addr() + obj.size();
    lock_write();
    object_mapped_ptr ptr(&obj);
    std::pair<Parent::iterator, bool> ret =
        Parent::insert(Parent::value_type(key, ptr));
    unlock();
    return ret.second;
}

object_mapped_ptr set_object_mapped::get_object(host_const_ptr addr) const
{
    object_mapped_ptr obj;
    lock_read();
    Parent::const_iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr >= i->second->addr());
    if(ret) {
        obj = i->second;
    }
    unlock();
    return obj;
}

bool set_object_mapped::remove(host_const_ptr addr)
{
    lock_write();
    Parent::iterator i = Parent::upper_bound(addr);
    bool ret = (i != end()) && (addr == i->second->addr());
    if(ret == true) erase(i);
    unlock();
    return ret;
}

set_object_mapped object_mapped::set_;

object_mapped::object_mapped(core::address_space_ptr aspace, size_t size) :
    size_(size),
    owner_(aspace)
{
    gmacError_t err;
    // Allocate memory (if necessary)
    addr_ = alloc(owner_, err);
    if(err != gmacSuccess) return;
    set_.add_object(*this);
    TRACE(LOCAL, "Creating Host Mapped Object @ %p) ", addr_.get_host_addr());
}


object_mapped::~object_mapped()
{
    if(addr_) free(owner_);
    TRACE(LOCAL, "Destroying Host Mapped Object @ %p", addr_.get_host_addr());
}

hal::ptr
object_mapped::get_device_addr(core::address_space_ptr current, host_const_ptr addr) const
{
    //ASSERTION(current == owner_);
    hal::ptr ret;
    if(addr_) {
        unsigned offset = unsigned(addr - addr_.get_host_addr());
        hal::ptr acceleratorAddr = getAccPtr(current);
        ret = acceleratorAddr + offset;
    }
    return ret;
}

hal::ptr
object_mapped::alloc(core::address_space_ptr aspace, gmacError_t &err)
{
    hal::ptr ret;
    //ret = Memory::map(NULL, size_, GMAC_PROT_READWRITE);
    ret = aspace->alloc_host_pinned(size_, err);
    return ret;
}

void
object_mapped::free(core::address_space_ptr aspace)
{
    //Memory::unmap(addr_, size_);
    aspace->free_host_pinned(addr_);
}

hal::ptr
object_mapped::getAccPtr(core::address_space_ptr aspace) const
{
#if 0
    gmacError_t err;
    return aspace->get_host_pinned_mapping(addr_.get_host_addr(), err);
#endif
    return hal::ptr();
}

}}
