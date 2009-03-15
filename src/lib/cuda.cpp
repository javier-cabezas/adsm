#include <gmac.h>
#include <loader.h>

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define PARAVER_NO_CUDA_OVERRIDE

#include <common/MemManager.h>
#include <common/debug.h>

SYM(cudaError_t, _cudaLaunch, const char *);

extern gmac::MemManager *memManager;

static void __attribute__((constructor)) gmacCudaInit(void)
{
	LOAD_SYM(_cudaLaunch, cudaLaunch);
}

cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

