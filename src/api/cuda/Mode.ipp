#ifndef __API_CUDA_MODE_IPP_H_
#define __API_CUDA_MODE_IPP_H_

#include "Context.h"

namespace gmac { namespace gpu {

inline
void Switch::in()
{
    dynamic_cast<Mode *>(Mode::current())->switchIn();
}

inline
void Switch::out()
{
    dynamic_cast<Mode *>(Mode::current())->switchOut();
}

inline
void Mode::switchIn()
{
    __mutex.lock();
#ifdef USE_MULTI_CONTEXT
    CUresult ret = cuCtxPushCurrent(__ctx);
    cfatal(ret != CUDA_SUCCESS, "Unable to switch to CUDA mode");
#else
    acc->switchIn();
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    cfatal(ret != CUDA_SUCCESS, "Unable to switch back from CUDA mode");
#else
    acc->switchOut();
#endif
    __mutex.unlock();
}

inline
gmacError_t Mode::bufferToDevice(gmac::IOBuffer *buffer, void *addr, size_t len)
{
    switchIn();
    gmacError_t ret = context->bufferToDevice(dynamic_cast<IOBuffer *>(buffer), addr, len);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToHost(gmac::IOBuffer *buffer, void *addr, size_t len)
{
    switchIn();
    gmacError_t ret = context->bufferToHost(dynamic_cast<IOBuffer *>(buffer), addr, len);
    switchOut();
    return ret;
}

inline
void Mode::call(dim3 Dg, dim3 Db, size_t shared, cudaStream_t tokens)
{
    switchIn();
    context->call(Dg, Db, shared, tokens);
    switchOut();
}

inline
void Mode::argument(const void *arg, size_t size, off_t offset)
{
    switchIn();
    context->argument(arg, size, offset);
    switchOut();
}


}}

#endif
