#ifndef GMAC_API_OPENCL_HPE_MODE_IMPL_H_
#define GMAC_API_OPENCL_HPE_MODE_IMPL_H_

#include "core/IOBuffer.h"

#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/hpe/Context.h"

namespace __impl { namespace opencl { namespace hpe {

inline
KernelList::KernelList() : gmac::util::Lock("KernelList")
{
}

inline
KernelList::~KernelList()
{
    Parent::const_iterator i;
    lock();
    for(i = Parent::begin(); i != Parent::end(); i++) delete *i;
    Parent::clear();
    unlock();
}

inline
void KernelList::insert(core::hpe::Kernel *kernel)
{
    lock();
    Parent::push_back(kernel);
    unlock();
}

inline
void Mode::switchIn()
{
}

inline
void Mode::switchOut()
{
}



inline gmacError_t
Mode::wait(core::hpe::KernelLaunch &launch)
{
    switchIn();
    error_ = contextMap_.waitForCall(launch);
    switchOut();

    return error_;
}


inline gmacError_t
Mode::wait()
{
    switchIn();
    error_ = contextMap_.waitForCall();
    switchOut();

    return error_;
}


inline
gmacError_t Mode::bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.get(), len);
    switchIn();
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.bufferToAccelerator(dst, buffer, len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.get(), buffer.addr(), len);
    switchIn();
    // Implement a function to remove these casts
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.acceleratorToBuffer(buffer, src, len, off);
    switchOut();
    return ret;
}

inline gmacError_t
Mode::prepareForCall()
{
    switchIn();
    gmacError_t ret = getAccelerator().syncCLstream(stream_);
    switchOut();
	return ret;
}

}}}

#endif
