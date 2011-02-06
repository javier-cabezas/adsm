#ifndef GMAC_API_OPENCL_MODE_IMPL_H_
#define GMAC_API_OPENCL_MODE_IMPL_H_

#include "core/Process.h"
#include "core/IOBuffer.h"

#include "Context.h"

namespace __impl { namespace opencl {

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
void KernelList::insert(core::Kernel *kernel)
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

inline core::KernelLaunch &
Mode::launch(gmacKernel_t name)
{
    KernelMap::iterator i = kernels_.find(name);
    Kernel *k = NULL;
    if (i == kernels_.end()) {
        k = dynamic_cast<Accelerator *>(acc_)->getKernel(name);
        ASSERTION(k != NULL);
        kernel(name, *k);
        kernelList_.insert(k);
    }
    else k = dynamic_cast<Kernel *>(i->second);
    switchIn();
    core::KernelLaunch &l = getCLContext().launch(*k);
    switchOut();
    return l;
}

inline gmacError_t
Mode::execute(core::KernelLaunch & launch)
{
    switchIn();
    gmacError_t ret = getContext().prepareForCall();
    if(ret == gmacSuccess) {
        trace::SetThreadState(THREAD_T(id_), trace::Running);
        ret = getAccelerator().execute(dynamic_cast<KernelLaunch &>(launch));
    }
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), dst.base_, len);
    switchIn();
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.bufferToAccelerator(dst, buffer, len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", src.base_, buffer.addr(), len);
    switchIn();
    // Implement a function to remove these casts
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.acceleratorToBuffer(buffer, src, len, off);
    switchOut();
    return ret;
}


inline Mode &
Mode::getCurrent()
{
    return static_cast<Mode &>(core::Mode::getCurrent());
}


inline Accelerator &
Mode::getAccelerator()
{
    return *static_cast<Accelerator *>(acc_);
}

inline
gmacError_t Mode::call(cl_uint workDim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize)
{
    switchIn();
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.call(workDim, globalWorkOffset, globalWorkSize, localWorkSize);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::argument(const void *arg, size_t size, unsigned index)
{
    switchIn();
    Context &ctx = getCLContext();
    gmacError_t ret = ctx.argument(arg, size, index);
    switchOut();
    return ret;
}

inline gmacError_t
Mode::eventTime(uint64_t &t, cl_event start, cl_event end)
{
    switchIn();
    gmacError_t ret = getAccelerator().timeCLevents(t, start, end);
    switchOut();
    return ret; 
}

}}

#endif
