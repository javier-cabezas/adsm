#ifndef GMAC_API_OPENCL_ACCELERATOR_IMPL_H_
#define GMAC_API_OPENCL_ACCELERATOR_IMPL_H_

#include <CL/cl.h>

#include "util/Logger.h"
#include "trace/Tracer.h"

#include "IOBuffer.h"

namespace __impl { namespace opencl {

inline CommandList::~CommandList()
{
    lockWrite();
    Parent::iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++)
        clReleaseCommandQueue(*i);
    Parent::clear();
    unlock();
}

inline void CommandList::add(cl_command_queue stream)
{
    lockWrite();
    Parent::push_back(stream);
    unlock();
}

inline void CommandList::remove(cl_command_queue stream)
{
    lockWrite();
    Parent::remove(stream);
    unlock();
}

inline cl_command_queue &CommandList::front()
{
    lockRead();
    cl_command_queue &ret = Parent::front();
    unlock();
    return ret;
}

inline cl_int CommandList::sync() const
{
    cl_int ret = CL_SUCCESS;
    lockRead();
    Parent::const_iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++) {
        if((ret = clFinish(*i)) != CL_SUCCESS) break;
    }
    unlock();
    return ret;
}


inline HostMap::~HostMap()
{
    // There is a race condition with the OpenCL thread. If the OpenCL
    // thread finishes before the main thread, OpenCL calls will access
    // to freed memory
#if 0
    lockWrite();
    Parent::iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++) {
        clReleaseMemObject(i->second);
    }
    Parent::clear();
    unlock();
#endif
}

inline void HostMap::insert(hostptr_t host, cl_mem acc)
{
    lockWrite();
    Parent::insert(Parent::value_type(host, acc));
    unlock();
}

inline void HostMap::remove(hostptr_t host)
{
    lockWrite();
    Parent::iterator i = Parent::find(host);
    if(i != Parent::end()) {
        Parent::erase(i);
    }
    unlock();
}

inline cl_mem HostMap::translate(hostptr_t host) const
{
    cl_mem ret(NULL);
    lockRead();
    Parent::const_iterator i = Parent::find(host);
    if(i != Parent::end()) ret = i->second;
    ASSERTION(i != Parent::end());
    unlock();
    return ret;
}

inline cl_device_id
Accelerator::device() const
{
    return device_;
}


inline
gmacError_t Accelerator::execute(KernelLaunch &launch)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL,"Executing KernelLaunch");
    gmacError_t ret = launch.execute();
    if (ret == gmacSuccess) trace::SetThreadState(THREAD_T(id_), trace::Running);
    trace::ExitCurrentFunction();
    return ret;
}

}}

#endif
