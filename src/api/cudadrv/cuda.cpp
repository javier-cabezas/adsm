#include <debug.h>
#include <order.h>

#include "GPU.h"
#include "GPUContext.h"

#include <gmac/init.h>
#include <kernel/System.h>

#include <cuda.h>

#include <string>
#include <list>

static unsigned nextGPU = 0;

namespace gmac {

gmac::Context *createContext()
{
	assert(sys != NULL);
	nextGPU = nextGPU % sys->getNumberOfAccelerators();
	gmac::GPU *gpu = dynamic_cast<gmac::GPU *>(sys->accelerator(nextGPU));
	return new gmac::GPUContext(*gpu);
}

void apiInit(void)
{
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
		sys->addAccelerator(new gmac::GPU(i, cuDev));
	}

	contextInit();
}

}
