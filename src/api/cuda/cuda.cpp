#include <order.h>

#include "Accelerator.h"
#include "Context.h"

#include <gmac/init.h>
#include <kernel/Process.h>

#include <cuda.h>

#include <string>
#include <list>

#include <cuda_runtime_api.h>

namespace gmac {

static bool initialized = false;

void apiInit(void)
{
	if(initialized)
		util::Logger::fatal("GMAC double initialization not allowed");

	util::Logger::ASSERTION(proc != NULL);
	util::Logger::TRACE("Initializing CUDA Driver API");
	if(cuInit(0) != CUDA_SUCCESS)
		util::Logger::fatal("Unable to init CUDA");

	int devCount = 0;
	int devRealCount = 0;

	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS)
		util::Logger::fatal("Error getting CUDA-enabled devices");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		CUdevice cuDev;
		int attr;
		if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
			util::Logger::fatal("Unable to access CUDA device");
#if CUDART_VERSION >= 2020
		if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
			util::Logger::fatal("Unable to access CUDA device");
		if(attr != CU_COMPUTEMODE_PROHIBITED) {
			proc->addAccelerator(new gmac::gpu::Accelerator(i, cuDev));
			devRealCount++;
		}
#else
        proc->addAccelerator(new gmac::gpu::Accelerator(i, cuDev));
        devRealCount++;
#endif
	}

	if(devRealCount == 0)
		util::Logger::fatal("No CUDA-enabled devices found");

	initialized = true;
}

}
