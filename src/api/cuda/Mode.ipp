#ifndef __API_CUDA_MODE_IPP_H_
#define __API_CUDA_MODE_IPP_H_

#include "Context.h"

namespace gmac { namespace cuda {

inline
void Mode::switchIn()
{
#ifdef USE_MULTI_CONTEXT
    acc->setCUcontext(&_cudaCtx);
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    acc->setCUcontext(NULL);
#endif
}

inline Context *
Mode::context()
{
    return dynamic_cast<Context *>(_context);
}

inline const Context *
Mode::context() const
{
    return dynamic_cast<Context *>(_context);
}

inline gmacError_t
Mode::execute(gmac::KernelLaunch * launch)
{
    switchIn();
    gmacError_t ret = acc->execute(dynamic_cast<KernelLaunch *>(launch));
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToDevice(void *dst, gmac::IOBuffer *buffer, size_t len, off_t off)
{
    switchIn();
    gmacError_t ret = context()->bufferToDevice(dst, dynamic_cast<IOBuffer *>(buffer), len, off);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::deviceToBuffer(gmac::IOBuffer *buffer, const void * src, size_t len, off_t off)
{
    switchIn();
    gmacError_t ret = context()->deviceToBuffer(dynamic_cast<IOBuffer *>(buffer), src, len, off);
    switchOut();
    return ret;
}

inline
void Mode::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    switchIn();
    context()->call(Dg, Db, shared, tokens);
    switchOut();
}

inline
void Mode::argument(const void *arg, size_t size, off_t offset)
{
    switchIn();
    context()->argument(arg, size, offset);
    switchOut();
}

inline Mode *
Mode::current()
{
    Mode *mode = static_cast<Mode *>(gmac::Mode::key.get());
    if(mode == NULL) mode = static_cast<Mode *>(proc->create());
    gmac::util::Logger::ASSERTION(mode != NULL);
    return mode;
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

}}

#endif
