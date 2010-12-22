#include "config/order.h"

#include "Accelerator.h"
#include "Mode.h"

#include "gmac/init.h"
#include "core/Process.h"

#include <cuda.h>

#include <string>
#include <list>

namespace __impl { namespace core {

static bool initialized = false;

void apiInit(void)
{
    core::Process &proc = core::Process::getInstance();
	if(initialized)
		FATAL("GMAC double initialization not allowed");

	TRACE(GLOBAL, "Initializing CUDA Driver API");
	if(cuInit(0) != CUDA_SUCCESS)
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
			proc.addAccelerator(new gmac::cuda::Accelerator(i, cuDev));
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
    cuda::Accelerator::init();

	initialized = true;
}

}}
