#include "config/common.h"

#include "coherence_domain.h"
#include "device.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; if(r == CUDA_ERROR_INVALID_HANDLE) abort(); break


namespace __impl { namespace hal {

gmacError_t
init_platform()
{
    static bool initialized = false;

    gmacError_t ret = gmacSuccess;

    if (initialized == false) {
        TRACE(GLOBAL, "Initializing CUDA Driver API");
        if (cuInit(0) != CUDA_SUCCESS) {
            FATAL("Unable to init CUDA");
        }
        initialized = true;
    } else {
        FATAL("Double HAL platform initialization");
    }

    return ret;
}

cuda::map_context_repository Modules_("map_context_repository");

std::list<cuda::device *>
init_devices()
{
    static bool initialized = false;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL device initialization");
    }

    std::list<cuda::device *> devices;

    int devCount = 0;
    int devRealCount = 0;
    
    CUresult err = cuDeviceGetCount(&devCount);
    if(err != CUDA_SUCCESS)
        FATAL("Error getting CUDA-enabled devices");

    TRACE(GLOBAL, "Found %d CUDA capable devices", devCount);

    // Add accelerators to the system
    for(int i = 0; i < devCount; i++) {
        CUdevice cuDev;
        if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");
#if CUDA_VERSION >= 2020
        int attr = 0;
        if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
            FATAL("Unable to access CUDA device");
        if(attr != CU_COMPUTEMODE_PROHIBITED) {
            cuda::coherence_domain *coherenceDomain = new cuda::coherence_domain();
            // TODO: detect coherence domain correctly. Now using one per device.
            cuda::device *dev = new cuda::device(cuDev, *coherenceDomain);
            devices.push_back(dev);
            devRealCount++;
        }
#else
        cuda::device *dev = new cuda::device(cuDev, *coherenceDomain);
        devices.push_back(dev);
        devRealCount++;
#endif
    }

    if(devRealCount == 0)
        MESSAGE("No CUDA-enabled devices found");

    return devices;
}

}}

namespace __impl { namespace hal { namespace cuda { 

gmacError_t error(CUresult err)
{
    gmacError_t error = gmacSuccess;
    switch(err) {
        __GMAC_ERROR(CUDA_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CUDA_ERROR_INVALID_VALUE, gmacErrorInvalidValue);
        __GMAC_ERROR(CUDA_ERROR_OUT_OF_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NOT_INITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_DEINITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoAccelerator);
        __GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidAccelerator);
        __GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidAcceleratorFunction);
        __GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidAcceleratorFunction);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_ACQUIRED, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_FILE_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_INVALID_HANDLE, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_READY, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_FAILED, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_UNKNOWN, gmacErrorUnknown);
        default: error = gmacErrorUnknown;
    }
    return error;
}

_event_t *
context_t::get_new_event(bool async,_event_t::type t)
{
    _event_t *ret = queueEvents_.pop();
    if (ret == NULL) {
        ret = new _event_t(async, t, *this);
    } else {
        ret->reset(async, t);
    }

    return ret;
}

void
context_t::dispose_event(_event_t &event)
{
    queueEvents_.push(event);
}

context_t::context_t(CUcontext ctx, device &dev) :
    Parent(ctx, dev)
{
    TRACE(LOCAL, "Creating context: %p", (*this)());
}

accptr_t
context_t::alloc(size_t count, gmacError_t &err)
{
    set();

    CUdeviceptr devPtr = 0;
    CUresult res = cuMemAlloc(&devPtr, count);

    err = cuda::error(res);

    return accptr_t(devPtr);
}

const code_repository &
context_t::get_code_repository()
{
    code_repository *repository;
    map_context_repository::iterator it = Modules_.find(this);
    if (it == Modules_.end()) {
        set();

        repository = module_descriptor::create_modules();
        Modules_.insert(map_context_repository::value_type(this, repository));
    } else {
        repository = it->second;
    }

    return *repository;
}

stream_t::stream_t(CUstream stream, context_t &context) :
    Parent(stream, context)
{
    TRACE(LOCAL, "Creating stream: %p", (*this)());
}

gmacError_t
stream_t::sync()
{
    get_context().set(); 

    TRACE(LOCAL, "Waiting for stream: %p", (*this)());
    CUresult ret = cuStreamSynchronize((*this)());

    return cuda::error(ret);
}

void
_event_t::reset(bool async, type t)
{
    isAsynchronous_ = async;
    type_ = t;
    err_ = gmacSuccess;
    synced_ = false;
    state_ = None;
}

_event_t::state
_event_t::get_state()
{
    if (state_ != End) {
        get_stream().get_context().set();

        CUresult res = cuEventQuery(eventEnd_);

        if (res == CUDA_ERROR_NOT_READY) {
            state_ = Queued;
        } else if (res == CUDA_SUCCESS) {
            state_ = End;
        }
    }

    return state_;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
