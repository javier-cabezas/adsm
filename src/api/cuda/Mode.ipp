#ifndef GMAC_API_CUDA_MODE_IPP_H_
#define GMAC_API_CUDA_MODE_IPP_H_

#include "core/Process.h"

#include "Context.h"

namespace gmac { namespace cuda {

inline
void Mode::switchIn()
{
#ifdef USE_MULTI_CONTEXT
    accelerator().setCUcontext(&_cudaCtx);
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    accelerator().setCUcontext(NULL);
#endif
}

inline gmacError_t
Mode::execute(gmac::KernelLaunch & launch)
{
    switchIn();
    gmacError_t ret = accelerator().execute(dynamic_cast<KernelLaunch &>(launch));
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToDevice(void *dst, gmac::IOBuffer &buffer, size_t len, off_t off)
{
    util::Logger::trace("Copy %p to device %p (%zd bytes)", buffer.addr(), dst, len);
    switchIn();
    gmacError_t ret = Context::current().bufferToDevice(dst, dynamic_cast<IOBuffer &>(buffer), len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::deviceToBuffer(gmac::IOBuffer &buffer, const void * src, size_t len, off_t off)
{
    util::Logger::trace("Copy %p to host %p (%zd bytes)", src, buffer.addr(), len);
    switchIn();
    gmacError_t ret = Context::current().deviceToBuffer(dynamic_cast<IOBuffer &>(buffer), src, len, off);
    switchOut();
    return ret;
}

inline
void Mode::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    switchIn();
    Context::current().call(Dg, Db, shared, tokens);
    switchOut();
}

inline
void Mode::argument(const void *arg, size_t size, off_t offset)
{
    switchIn();
    Context::current().argument(arg, size, offset);
    switchOut();
}

inline Mode &
Mode::current()
{
    return static_cast<Mode &>(gmac::Mode::current());
}

#ifdef USE_VM
inline CUdeviceptr
Mode::dirtyBitmapDevPtr() const
{
    return _bitmapDevPtr;
}

inline CUdeviceptr
Mode::dirtyBitmapShiftPageDevPtr() const
{
    return _bitmapShiftPageDevPtr;
}

#ifdef BITMAP_BIT
inline CUdeviceptr
Mode::dirtyBitmapShiftEntryDevPtr() const
{
    return _bitmapShiftEntryDevPtr;
}
#endif
#endif

inline Accelerator &
Mode::accelerator()
{
    return *static_cast<Accelerator *>(acc_);
}

}}

#endif
