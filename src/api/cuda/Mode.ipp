#ifndef __API_CUDA_MODE_IPP_H_
#define __API_CUDA_MODE_IPP_H_

namespace gmac { namespace gpu {

inline
void Mode::switchIn()
{
#ifdef USE_MULTI_CONTEXT
    __mutex.lock();
    CUresult ret = cuCtxPushCurrent(__ctx);
    cfatal(ret != CUDA_SUCCESS, "Unable to switch to CUDA mode");
#endif
}

inline
void Mode::switchOut()
{
#ifdef USE_MULTI_CONTEXT
    CUcontext tmp;
    CUresult ret = cuCtxPopCurrent(&tmp);
    __mutex.unlock();
    cfatal(ret != CUDA_SUCCESS, "Unable to switch back from CUDA mode");
#endif
}

inline
IOBuffer *Mode::getIOBuffer(size_t size)
{
    IOBuffer *buffer = new IOBuffer(this, size);
    if(buffer->addr() == NULL) { delete buffer; return NULL; }
    return buffer;
}

inline
gmacError_t Mode::bufferToDevice(IOBuffer *buffer, void *addr, size_t len)
{
    switchIn();
    gmacError_t ret = context->bufferToDevice(buffer, addr, len);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::waitDevice()
{
    switchIn();
    gmacError_t ret = context->waitDevice();
    switchOut();
    return ret;
}

inline
gmacError_t Mode::bufferToHost(IOBuffer *buffer, void *addr, size_t len)
{
    switchIn();
    gmacError_t ret = context->bufferToHost(buffer, addr, len);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::waitHost()
{
    switchIn();
    gmacError_t ret = context->waitHost();
    switchOut();
    return ret;
}

}}

#endif
