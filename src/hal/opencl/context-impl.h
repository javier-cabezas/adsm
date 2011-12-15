#ifndef GMAC_HAL_OPENCL_CONTEXT_IMPL_H_
#define GMAC_HAL_OPENCL_CONTEXT_IMPL_H_

#include "trace/logger.h"

namespace __impl { namespace hal { namespace opencl {

inline
buffer_t::buffer_t(hostptr_t addr, cl_mem devPtr, size_t size, context_t &context) :
    Parent(size, context),
    addr_(addr),
    devPtr_(devPtr)
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
    return ptr_t(devPtr_, &get_context());
}

inline
ptr_t
context_t::get_device_addr_from_pinned(hostptr_t addr)
{
    FATAL("NOT SUPPORTED IN OPENCL");

    return ptr_t();
}

}}} /* GMAC_HAL_OPENCL_CONTEXT_IMPL_H_ */

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
