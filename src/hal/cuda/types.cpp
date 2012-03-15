#include <sys/sysinfo.h>

#include "config/common.h"

#include "phys/aspace.h"
#include "phys/processing_unit.h"
#include "phys/platform.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; if(r == CUDA_ERROR_INVALID_HANDLE) abort(); break

namespace __impl { namespace hal {

static bool
hal_initialized = false;

static
cuda::phys::list_platform platforms;

gmacError_t
init()
{
    gmacError_t ret = gmacSuccess;

    if (hal_initialized == false) {
        TRACE(GLOBAL, "Initializing CUDA Driver API");
        if (cuInit(0) != CUDA_SUCCESS) {
            FATAL("Unable to init CUDA");
        }
        hal_initialized = true;
    } else {
        FATAL("Double HAL platform initialization");
    }

    return ret;
}

gmacError_t
fini()
{
    if (hal_initialized == false) {
        FATAL("HAL not initialized");
    }

    gmacError_t ret = gmacSuccess;

    hal_initialized = false;

    for (cuda::phys::platform *p : platforms) {
        delete p;
    }

    platforms.clear();

    return ret;
}

namespace phys {

cuda::phys::list_platform
get_platforms()
{
    if (hal_initialized == false) {
        FATAL("HAL not initialized");
    }

    if (platforms.size() == 0) {
        /////////////////////
        // Backend devices //
        /////////////////////
        cuda::phys::platform *p = new cuda::phys::platform();

        int devCount = 0;
        int devRealCount = 0;

        CUresult err = cuDeviceGetCount(&devCount);
        if(err != CUDA_SUCCESS)
            FATAL("Error getting CUDA-enabled devices");

        TRACE(GLOBAL, "Found %d CUDA capable devices", devCount);

        // Add accelerators to the system
        for (int i = 0; i < devCount; i++) {
            CUdevice cuDev;
            if (cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
                FATAL("Unable to access CUDA device");
            // TODO: create a context to retrieve information about the memory
            cuda::phys::memory *mem = new cuda::phys::memory(128 * 1024 * 1024);

            cuda::phys::aspace::set_memory memories;
            memories.insert(mem);

            cuda::phys::processing_unit::set_memory_connection connections;
            cuda::phys::processing_unit::memory_connection connection(*mem, 0);
            connections.insert(connection);

            cuda::phys::aspace *aspace = new cuda::phys::aspace(memories);
            cuda::phys::processing_unit::set_aspace aspaces;
            aspaces.insert(aspace);
#if CUDA_VERSION >= 2020
            int attr = 0;
            if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
                FATAL("Unable to access CUDA device");
            if (attr != CU_COMPUTEMODE_PROHIBITED) {
                cuda::phys::processing_unit *pUnit = new cuda::phys::processing_unit(cuDev, *p);
                p->add_processing_unit(*pUnit);
                devRealCount++;
            }
#else
            cuda::processing_unit *pUnit = new cuda::gpu(cuDev, *p);
            p->add_processing_unit(*pUnit);
            devRealCount++;
#endif
        }

        if (devRealCount == 0)
            MESSAGE("No CUDA-enabled devices found");

        /////////////////////
        // Backend devices //
        /////////////////////
#if 0
        {
            for (int i = 0; i < get_nprocs(); ++i) {
                p->add_device(*new cuda::cpu(*p, *cpuDomain));
            }
        }
#endif

        platforms.push_back(p);
    }

    return platforms;
}

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
        __GMAC_ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoDevice);
        __GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidDevice);
        __GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidDeviceFunction);
        __GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidDeviceFunction);
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

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
