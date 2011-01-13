#ifndef GMAC_API_CUDA_MODE_IMPL_H_
#define GMAC_API_CUDA_MODE_IMPL_H_

#include "core/Process.h"
#include "core/IOBuffer.h"

#include "Context.h"

namespace __impl { namespace cuda {

inline
void Mode::switchIn()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().setCUcontext(&cudaCtx_);
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    getAccelerator().setCUcontext(NULL);
#endif
}

inline
core::KernelLaunch &Mode::launch(gmacKernel_t kernel)
{
    KernelMap::iterator i = kernels_.find(kernel);
    ASSERTION(i != kernels_.end());
    core::Kernel * k = i->second;
    switchIn();
    core::KernelLaunch &l = getContext().launch(*k);
    switchOut();

    return l;
}

inline gmacError_t
Mode::execute(core::KernelLaunch & launch)
{
    switchIn();
    gmacError_t ret = getAccelerator().execute(dynamic_cast<KernelLaunch &>(launch));
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to device %p ("FMT_SIZE" bytes)", buffer.addr(), (void *) dst, len);
    switchIn();
    Context &ctx = dynamic_cast<Context &>(getContext());
    gmacError_t ret = ctx.bufferToAccelerator(dst, buffer, len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t src, size_t len, size_t off)
{
    TRACE(LOCAL,"Copy %p to host %p ("FMT_SIZE" bytes)", (void *) src, buffer.addr(), len);
    switchIn();
    // Implement a function to remove these casts
    Context &ctx = dynamic_cast<Context &>(getContext());
    gmacError_t ret = ctx.acceleratorToBuffer(buffer, src, len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    switchIn();
    Context &ctx = dynamic_cast<Context &>(getContext());
    gmacError_t ret = ctx.call(Dg, Db, shared, tokens);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::argument(const void *arg, size_t size, off_t offset)
{
    switchIn();
    Context &ctx = dynamic_cast<Context &>(getContext());
    gmacError_t ret = ctx.argument(arg, size, offset);
    switchOut();
    return ret;
}

inline Mode &
Mode::current()
{
    return static_cast<Mode &>(core::Mode::current());
}

#ifdef USE_VM
inline CUdeviceptr
Mode::dirtyBitmapAccPtr() const
{
    return bitmapAccPtr_;
}

inline CUdeviceptr
Mode::dirtyBitmapShiftPageAccPtr() const
{
    return bitmapShiftPageAccPtr_;
}

#endif

inline Accelerator &
Mode::getAccelerator()
{
    return *static_cast<Accelerator *>(acc_);
}

}}

#endif
