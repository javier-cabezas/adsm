#ifndef GMAC_API_OPENCL_KERNEL_IMPL_H_
#define GMAC_API_OPENCL_KERNEL_IMPL_H_

#include "util/Logger.h"
#include "gmac/init.h"

namespace __impl { namespace opencl {

inline
Argument::Argument(const void * ptr, size_t size, unsigned index) :
    ptr_(ptr), size_(size), index_(index)
{
}

inline
Kernel::Kernel(const core::hpe::KernelDescriptor & k, cl_kernel kernel) :
    gmac::core::hpe::Kernel(k), f_(kernel)
{
}

inline
Kernel::~Kernel()
{
    if(gmacFini__ < 0) clReleaseKernel(f_);
}

inline
KernelLaunch *
Kernel::launch(KernelConfig &c)
{
    KernelLaunch * l = new KernelLaunch(*this, c);
    return l;
}

inline
KernelConfig::KernelConfig() :
    argsSize_(0),
    globalWorkOffset_(NULL),
    globalWorkSize_(NULL),
    localWorkSize_(NULL)
{
}

inline
KernelConfig::~KernelConfig()
{
    if (globalWorkOffset_) delete [] globalWorkOffset_;
    if (globalWorkSize_) delete [] globalWorkSize_;
    if (localWorkSize_) delete [] localWorkSize_;
}

inline
void
KernelConfig::setArgument(const void * arg, size_t size, unsigned index)
{
    ASSERTION(argsSize_ + size < KernelConfig::StackSize_);

    ::memcpy(&stack_[argsSize_], arg, size);
    push_back(Argument(&stack_[argsSize_], size, index));
    argsSize_ += size;
}

inline
KernelLaunch::KernelLaunch(const Kernel & k, const KernelConfig & c) :
    core::hpe::KernelLaunch(),
    KernelConfig(c),
    f_(k.f_)
{
    clRetainKernel(f_);
}

inline
KernelLaunch::~KernelLaunch()
{
    clReleaseKernel(f_);
}



}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
