#include "Context.h"

#include <config.h>

#include <memory/Manager.h>
#include <gmac/init.h>

namespace gmac { namespace gpu {

Context::AddressMap Context::hostMem;
void * Context::FatBin;


    _bufferPageLockedSize = paramBufferPageLockedSize * paramPageSize;
#if CUDART_VERSION >= 2020
    ret = cuMemHostAlloc(&_bufferPageLocked, _bufferPageLockedSize, CU_MEMHOSTALLOC_PORTABLE);
#else
    ret = cuMemAllocHost(&_bufferPageLocked, _bufferPageLockedSize);
#endif
    if (ret == CUDA_SUCCESS) {
        trace("Using page locked memory: %zd\n", _bufferPageLockedSize);
    } else {
        fatal("Error %d: when allocating page-locked memory", ret);
    }
}

void
Context::finiStreams()
{
    CUresult ret;
    ret = cuStreamDestroy(streamLaunch);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamToDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamToHost);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuStreamDestroy(streamDevice);
    cfatal(ret == CUDA_SUCCESS, "Error destroying CUDA streams: %d", ret);
    ret = cuMemFreeHost(_bufferPageLocked);
    cfatal(ret == CUDA_SUCCESS, "Error freeing page-locked buffer: %d", ret);
}

Context::Context(Accelerator *gpu) :
    gmac::Context(gpu),
    _gpu(gpu),
    _call(dim3(0), dim3(0), 0, 0)
#ifdef USE_MULTI_CONTEXT
    , mutex(paraver::LockCtxLocal)
#endif
    , _pendingKernel(false)
    , _pendingToDevice(false)
    , _pendingToHost(false)
{
	setup();

    pushLock();
    setupStreams();
    _modules = ModuleDescriptor::createModules(*this);
    popUnlock();
}

Context::~Context()
{
    trace("Remove Accelerator context [%p]", this);
    ModuleVector::const_iterator m;
    for(m = _modules.begin(); m != _modules.end(); m++) {
        delete (*m);
    }
    _modules.clear();
    clearKernels();
}


}}
