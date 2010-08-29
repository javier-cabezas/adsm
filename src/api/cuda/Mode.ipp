#ifndef __API_CUDA_MODE_IPP_H_
#define __API_CUDA_MODE_IPP_H_

namespace gmac { namespace gpu {

inline void Switch::in()
{
    dynamic_cast<Mode *>(Mode::current())->switchIn();
}

inline void Switch::out()
{
    dynamic_cast<Mode *>(Mode::current())->switchOut();
}



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

}}

#endif
