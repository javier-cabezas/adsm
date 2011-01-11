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

inline void CommandList::sync() const
{
    lockRead();
    Parent::const_iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++)
        clFinish(*i);
    unlock();
}



inline HostMap::~HostMap()
{
    lockWrite();
    Parent::iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++) {
        clReleaseMemObject(i->second);
        free(i->first);
        Parent::erase(i);
    }
    unlock();
}

inline void HostMap::insert(hostptr_t *host, cl_mem acc)
{
    lockWrite();
    Parent::insert(Parent::value_type(host, acc));
    unlock();
}

inline void HostMap::remove(hostptr_t *host)
{
    lockWrite();
    Parent::iterator i = Parent::find(host);
    if(i != Parent::end()) {
        Parent::erase(i);
    }
    unlock();
}

inline accptr_t HostMap::translate(hostptr_t *host) const
{
    accptr_t ret(NULL);
    lockRead();
    Parent::const_iterator i = Parent::find(host);
    if(i != Parent::end()) ret.base_ = i->second;
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
    trace::ExitCurrentFunction();
    return ret;
}



}}

#endif
