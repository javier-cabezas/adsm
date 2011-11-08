#ifndef GMAC_HAL_CUDA_CONTEXT_IMPL_H_
#define GMAC_HAL_CUDA_CONTEXT_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace hal { namespace cuda {

inline
buffer_t::buffer_t(hostptr_t addr, size_t size, context_t &context) :
    Parent(size, context),
    addr_(addr)
{
}

inline
hostptr_t
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
gmacError_t
list_event::sync()
{
    gmacError_t ret = gmacSuccess;
    for (Parent::iterator it  = Parent::begin();
            it != Parent::end();
            it++) {
        ret = (*it).sync();
        if (ret != gmacSuccess) break;
    }

    return ret;
}

inline
void
context_t::set()
{
    CUresult ret = cuCtxSetCurrent(this->context_);
    ASSERTION(ret == CUDA_SUCCESS);
}

inline
ptr_t
context_t::get_device_addr_from_pinned(hostptr_t addr)
{
    ptr_t ret(0, NULL);
    set();

    CUdeviceptr ptr;
    CUresult res = cuMemHostGetDevicePointer(&ptr, addr, 0);
    if (res == CUDA_SUCCESS) {
        ret = ptr_t(ptr, this);
    }

    return ret;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
