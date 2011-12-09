#ifndef GMAC_HAL_OPENCL_CONTEXT_IMPL_H_
#define GMAC_HAL_OPENCL_CONTEXT_IMPL_H_

#include "util/Logger.h"

namespace __impl { namespace hal { namespace opencl {

inline
list_event::~list_event()
{
    for (Parent::iterator it  = Parent::begin();
                          it != Parent::end();
                          it++) {
        cl_int res = clReleaseEvent((*it)());
        ASSERTION(res == CL_SUCCESS);
    }
}

inline
gmacError_t
list_event::sync()
{
    cl_event *evs = get_event_array();

    cl_int res = clWaitForEvents(Parent::size(), evs);
    set_synced();

    return error(res);
}

inline
void
list_event::set_synced()
{
    for (Parent::iterator it  = Parent::begin();
            it != Parent::end();
            it++) {
        (*it).set_synced();
    }
}

inline
cl_event *
list_event::get_event_array()
{
    cl_event *ret = NULL;
    if (Parent::size() > 0) {
        ret = new cl_event[Parent::size()];
        int i = 0;
        for (Parent::iterator it = Parent::begin();
             it != Parent::end();
             it++) {
            ret[i++] = (*it)();
        }
    }

    return ret;
}

inline
size_t
list_event::size() const
{
    return Parent::size(); 
}

inline
void
list_event::add_event(event_t event) 
{
    locker::lock(*event);
    cl_event ev = event();
    cl_int res = clRetainEvent(ev);
    ASSERTION(res == CL_SUCCESS);
    Parent::push_back(event);
    locker::unlock(*event);
}

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
