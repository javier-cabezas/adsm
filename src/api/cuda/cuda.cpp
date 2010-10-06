#include <order.h>

#include "Accelerator.h"
#include "Mode.h"

#include "gmac/init.h"
#include "core/Process.h"

#include <cuda.h>

#include <string>
#include <list>

#include <cuda_runtime_api.h>

namespace gmac {

static bool initialized = false;

void apiInit(void)
{
    gmac::Process &proc = gmac::Process::getInstance();
	if(initialized)
		util::Logger::Fatal("GMAC double initialization not allowed");

	util::Logger::TRACE("Initializing CUDA Driver API");
	if(cuInit(0) != CUDA_SUCCESS)
		util::Logger::Fatal("Unable to init CUDA");

	int devCount = 0;
	int devRealCount = 0;

	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS)
		util::Logger::Fatal("Error getting CUDA-enabled devices");

    util::Logger::TRACE("Found %d CUDA capable devices", devCount);

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		CUdevice cuDev;
		int attr;
		if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
			util::Logger::Fatal("Unable to access CUDA device");
#if CUDART_VERSION >= 2020
		if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
			util::Logger::Fatal("Unable to access CUDA device");
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
		util::Logger::Fatal("No CUDA-enabled devices found");

    // Initialize the private per-thread variables
    gmac::cuda::Accelerator::init();

	initialized = true;
}

}
