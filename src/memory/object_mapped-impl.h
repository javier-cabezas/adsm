#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_

namespace __impl { namespace memory {

inline set_object_host_mapped::set_object_host_mapped() :
    gmac::util::lock_rw<set_object_host_mapped>("HostMappedSet")
{}

inline set_object_host_mapped::~set_object_host_mapped()
{}

#ifdef USE_OPENCL
inline gmacError_t
object_host_mapped::acquire(core::address_space_ptr current)
{
    gmacError_t ret = gmacSuccess;
#if 0
    ret = current->acquire(addr_);
#endif
    return ret;
}

inline gmacError_t
object_host_mapped::release(core::address_space_ptr current)
{
    gmacError_t ret = gmacSuccess;
#if 0
    ret = current->release(addr_);
#endif
    return ret;
}
#endif

inline hostptr_t object_host_mapped::addr() const
{
    return addr_.get_host_addr();
}

inline size_t object_host_mapped::size() const
{
    return size_;
}

inline void object_host_mapped::remove(hostptr_t addr)
{
    set_.remove(addr);
}

inline object_host_mapped_ptr object_host_mapped::get_object(const hostptr_t addr)
{
    return set_.get_object(addr);
}

}}

#endif
