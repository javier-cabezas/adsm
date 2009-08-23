#include <os/loader.h>
#include <order.h>

#include "GPU.h"
#include "Context.h"

#include <kernel/Process.h>

#include <cuda_runtime.h>
#include <cuda.h>


SYM(cudaError_t, __cudaLaunch, const char *);

static void __attribute__((constructor(INTERPOSE))) gmacCudaInit(void)
{
	LOAD_SYM(__cudaLaunch, cudaLaunch);
}

static unsigned nextGPU = 0;

namespace gmac {

void apiInit(void) 
{
	TRACE("Initializing CUDA Run-time API");

	int devCount = 0;
	if(cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0)
		FATAL("No CUDA-enable devices found");

	// Add accelerators to the system
	for(int i = 0; i < devCount; i++) {
		proc->accelerator(new gmac::GPU(i));
	}
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


