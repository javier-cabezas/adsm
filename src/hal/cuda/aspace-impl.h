#ifndef GMAC_HAL_CUDA_CONTEXT_IMPL_H_
#define GMAC_HAL_CUDA_CONTEXT_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace hal { namespace cuda {

inline aspace &
buffer_t::get_aspace()
{
    return reinterpret_cast<aspace &>(parent::get_aspace());
}

inline const aspace &
buffer_t::get_aspace() const
{
    return reinterpret_cast<const aspace &>(parent::get_aspace());
}

inline
buffer_t::buffer_t(host_ptr addr, size_t size, aspace &as) :
    parent(size, as),
    addr_(addr)
{
}

inline
host_ptr
buffer_t::get_addr()
{
    return addr_;
}

inline
hal::ptr
buffer_t::get_device_addr()
{
    return get_aspace().get_device_addr_from_pinned(addr_);
}

inline
void
aspace::set()
{
    CUresult ret = cuCtxSetCurrent(this->context_);
    ASSERTION(ret == CUDA_SUCCESS);
}

inline
hal::ptr
aspace::get_device_addr_from_pinned(host_ptr addr)
{
    hal::ptr ret;
    set();

    CUdeviceptr ptr;
    CUresult res = cuMemHostGetDevicePointer(&ptr, addr, 0);
    if (res == CUDA_SUCCESS) {
        ret = hal::ptr(hal::ptr::backend_ptr(ptr), this);
    }

    return ret;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
