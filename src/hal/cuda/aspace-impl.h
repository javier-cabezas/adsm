#ifndef GMAC_HAL_CUDA_CONTEXT_IMPL_H_
#define GMAC_HAL_CUDA_CONTEXT_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace hal { namespace cuda {

inline
buffer_t::buffer_t(host_ptr addr, size_t size, aspace &context) :
    Parent(size, context),
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
ptr_t
buffer_t::get_device_addr()
{
    return get_context().get_device_addr_from_pinned(addr_);
}

inline
void
aspace::set()
{
    CUresult ret = cuCtxSetCurrent(this->context_);
    ASSERTION(ret == CUDA_SUCCESS);
}

inline
ptr_t
aspace::get_device_addr_from_pinned(host_ptr addr)
{
    ptr_t ret;
    set();

    CUdeviceptr ptr;
    CUresult res = cuMemHostGetDevicePointer(&ptr, addr, 0);
    if (res == CUDA_SUCCESS) {
        ret = ptr_t(ptr_t::backend_ptr(ptr), this);
    }

    return ret;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
