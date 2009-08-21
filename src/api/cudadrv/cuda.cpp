#include <debug.h>
#include <order.h>

#include "GPU.h"
#include "Context.h"

#include <kernel/Process.h>

#include <cuda.h>

#include <string>
#include <list>

static unsigned nextGPU = 0;

namespace gmac {

void apiInit(void)
{
	assert(proc != NULL);
	TRACE("Initializing CUDA Driver API");
	if(cuInit(0) != CUDA_SUCCESS)
		FATAL("Unable to init CUDA");
}

void apiInitDevices(void)
{
	int devCount = 0;
	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS || devCount == 0)
		FATAL("No CUDA-enable devices found");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		CUdevice cuDev;
		if(cuDeviceGet(&cuDev, 0) != CUDA_SUCCESS)
			FATAL("Unable to access CUDA device");
		proc->addAccelerator(new gmac::GPU(i, cuDev));
	}
}

#if 0
Context *contextCreate(Accelerator *acc)
{
	GPU *gpu = dynamic_cast<GPU *>(acc);
	return new gpu::Context(*gpu);
}
#endif

}
