#include "gmac.h"

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <common/debug.h>

cudaError_t (*_cudaLaunch)(const char *) = NULL;

static void __attribute__((constructor)) gmacCudaInit(void)
{
	if((_cudaLaunch = (cudaError_t (*)(const char *))dlsym(RTLD_NEXT, "cudaLaunch")) == NULL)
		FATAL("cudaLaunch not found");
}

cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

