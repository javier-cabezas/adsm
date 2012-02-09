#ifndef GMAC_HAL_OPENCL_CONTEXT_IMPL_H_
#define GMAC_HAL_OPENCL_CONTEXT_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace hal { namespace opencl {

inline
buffer_t::buffer_t(host_ptr addr, cl_mem devPtr, size_t size, aspace &context) :
    Parent(size, context),
    addr_(addr),
    devPtr_(devPtr)
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
    return ptr_t(devPtr_, &get_context());
}

inline
ptr_t
aspace::get_device_addr_from_pinned(host_ptr addr)
{
    FATAL("NOT SUPPORTED IN OPENCL");

    return ptr_t();
}

}}} /* GMAC_HAL_OPENCL_CONTEXT_IMPL_H_ */

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
