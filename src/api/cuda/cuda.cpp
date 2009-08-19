#include <os/loader.h>
#include <order.h>

#include "GPU.h"
#include "GPUContext.h"

#include <gmac/gmac.h>
#include <gmac/init.h>

#include <kernel/System.h>

#include <cuda_runtime.h>
#include <cuda.h>


SYM(gmacError_t, __gmacLaunch, const char *);

static void __attribute__((constructor(INTERPOSE))) gmacCudaInit(void)
{
	LOAD_SYM(__gmacLaunch, cudaLaunch);
}

static unsigned nextGPU = 0;

namespace gmac {
Context *createContext()
{
	assert(sys != NULL);
	nextGPU = nextGPU % sys->getNumberOfAccelerators();
	GPU *gpu = dynamic_cast<gmac::GPU *>(sys->accelerator(nextGPU));
	return new gmac::GPUContext(*gpu);
}

void apiInit(void) 
{
	TRACE("Initializing CUDA Run-time API");

	int devCount = 0;
	if(cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
		FATAL("No CUDA-enable devices found");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		sys->addAccelerator(new gmac::GPU(i));
	}

	contextInit();
}
}

#ifdef __cplusplus
extern "C" {
#endif


extern gmacError_t gmacLaunch(const char *);
cudaError_t cudaLaunch(const char *symbol)
{
	return (cudaError_t)gmacLaunch(symbol);
}

#ifdef __cplusplus
}
#endif


