#include <debug.h>
#include <order.h>

#include "GPU.h"
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
	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS || devCount == 0)
		FATAL("No CUDA-enable devices found");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		CUdevice cuDev;
		if(cuDeviceGet(&cuDev, 0) != CUDA_SUCCESS)
			FATAL("Unable to access CUDA device");
		proc->accelerator(new gmac::GPU(i, cuDev));
	}

	initialized = true;
}

}
