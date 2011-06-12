#include <cuda.h>

#include <string>
#include <list>


#include "api/cuda/hpe/Accelerator.h"
#include "api/cuda/hpe/Mode.h"

#include "config/order.h"
#include "core/Process.h"

#include "hpe/init.h"

#include "util/loader.h"

static bool initialized = false;

void GMAC_API CUDA(gmac::core::hpe::Process &proc)
{
    TRACE(GLOBAL, "Initializing CUDA Driver API");
    if(initialized == false && cuInit(0) != CUDA_SUCCESS)
        FATAL("Unable to init CUDA");

    int devCount = 0;
    int devRealCount = 0;

    if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS)
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
            gmac::cuda::hpe::Accelerator *accelerator = new gmac::cuda::hpe::Accelerator(i, cuDev);
            CFATAL(accelerator != NULL, "Error allocating resources for the accelerator");
            proc.addAccelerator(*accelerator);
            devRealCount++;
        }
#else
        proc.addAccelerator(new gmac::cuda::Accelerator(i, cuDev));
        devRealCount++;
#endif
    }

    if(devRealCount == 0)
        FATAL("No CUDA-enabled devices found");

    // Initialize the private per-thread variables
    gmac::cuda::hpe::Accelerator::init();


    initialized = true;
}
