#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_IMPL_H_

namespace __impl { namespace memory {

inline set_object_mapped::set_object_mapped() :
    gmac::util::lock_rw<set_object_mapped>("HostMappedSet")
{}

inline set_object_mapped::~set_object_mapped()
{}

#ifdef USE_OPENCL
inline gmacError_t
object_mapped::acquire(core::address_space_ptr current)
{
    gmacError_t ret = gmacSuccess;
#if 0
    ret = current->acquire(addr_);
#endif
    return ret;
}

inline gmacError_t
object_mapped::release(core::address_space_ptr current)
{
    gmacError_t ret = gmacSuccess;
#if 0
    ret = current->release(addr_);
#endif
    return ret;
}
#endif

inline hostptr_t object_mapped::addr() const
{
    return addr_.get_host_addr();
}

inline size_t object_mapped::size() const
{
    return size_;
}

inline void object_mapped::remove(hostptr_t addr)
{
    set_.remove(addr);
}

inline object_mapped_ptr object_mapped::get_object(const hostptr_t addr)
{
    return set_.get_object(addr);
}

}}

#endif
