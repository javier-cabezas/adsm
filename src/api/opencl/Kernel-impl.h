#ifndef GMAC_API_OPENCL_KERNEL_IMPL_H_
#define GMAC_API_OPENCL_KERNEL_IMPL_H_

#include "util/Logger.h"
#include "gmac/init.h"

namespace __impl { namespace opencl {

inline
Argument::Argument() :
    size_(0)
{
}

inline void
Argument::setArgument(const void * ptr, size_t size)
{
    size_ = size;

    ::memcpy(stack_, ptr, size);
}

inline
Kernel::Kernel(const core::KernelDescriptor & k, cl_kernel kernel) :
    gmac::core::Kernel(k), f_(kernel) //added
{
    cl_int ret = clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &nArgs_, NULL);
    ASSERTION(ret == CL_SUCCESS);
}

inline
Kernel::~Kernel()
{
    if(gmacFini__ < 0) clReleaseKernel(f_);
}

inline
KernelLaunch *
Kernel::launch(core::Mode &mode, cl_command_queue stream)
{
    KernelLaunch * l = new KernelLaunch(mode, *this, stream);
    return l;
}

inline
KernelConfig::KernelConfig(unsigned nArgs) :
    ArgsVector(nArgs),
    workDim_(0),
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
KernelConfig::setConfiguration(cl_uint work_dim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize)
{
    if (workDim_ < work_dim) {
        if (globalWorkOffset_) {
            delete [] globalWorkOffset_;
            globalWorkOffset_ = NULL;
        }
        if (globalWorkSize_) {
            delete [] globalWorkSize_;
            globalWorkSize_ = NULL;
        }
        if (localWorkSize_) {
            delete [] localWorkSize_;
            localWorkSize_ = NULL;
        }

        if (globalWorkOffset) {
            globalWorkOffset_ = new size_t[work_dim];
        }
        if (globalWorkSize) {
            globalWorkSize_ = new size_t[work_dim];
        }
        if (localWorkSize) {
            localWorkSize_ = new size_t[work_dim];
        }
    }

    workDim_ = work_dim;

    for (unsigned i = 0; i < workDim_; i++) {
        if (globalWorkOffset) globalWorkOffset_[i] = globalWorkOffset[i];
        if (globalWorkSize) globalWorkSize_[i] = globalWorkSize[i];
        if (localWorkSize) localWorkSize_[i] = localWorkSize[i];
    }
}

inline
void
KernelConfig::setArgument(const void * arg, size_t size, unsigned index)
{
    ArgsVector::at(index).setArgument(arg, size);
}

inline
KernelLaunch::KernelLaunch(core::Mode &mode, const Kernel & k, cl_command_queue stream) :
    __impl::core::KernelLaunch(mode),
    KernelConfig(k.nArgs_),
    f_(k.f_),
    stream_(stream)
{
    clRetainKernel(f_);
}

inline
KernelLaunch::~KernelLaunch()
{
    clReleaseKernel(f_);
}

inline
cl_event
KernelLaunch::getCLEvent()
{
    return event_;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
