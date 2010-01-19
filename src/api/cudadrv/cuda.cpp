#include <debug.h>
#include <order.h>

#include "Accelerator.h"
#include "Context.h"

#include <kernel/Process.h>

#include <cuda.h>

#include <string>
#include <list>


namespace gmac {

static bool initialized = false;

void apiInit(void)
{
	if(initialized)
		FATAL("GMAC double initialization not allowed");

	assert(proc != NULL);
	TRACE("Initializing CUDA Driver API");
	if(cuInit(0) != CUDA_SUCCESS)
		FATAL("Unable to init CUDA");

	int devCount = 0;
	int devRealCount = 0;

	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS)
		FATAL("Error getting CUDA-enabled devices");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		CUdevice cuDev;
		int attr;
		if(cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
			FATAL("Unable to access CUDA device");
		if(cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
			FATAL("Unable to access CUDA device");
		if(attr != CU_COMPUTEMODE_PROHIBITED) {
			proc->accelerator(new gmac::gpu::Accelerator(i, cuDev));
			devRealCount++;
		}
	}

	if(devRealCount == 0)
		FATAL("No CUDA-enabled devices found");

	initialized = true;
}

}
