#include "gmac.h"

#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <common/MemManager.h>
#include <common/debug.h>

typedef cudaError_t (*cudaLaunch_t)(const char *);
typedef cudaError_t (*cudaConfigureCall_t)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t (*cudaSetupArgument_t)(void *, size_t, size_t);

cudaLaunch_t _cudaLaunch = NULL;
cudaConfigureCall_t _cudaConfigureCall = NULL;
cudaSetupArgument_t _cudaSetupArgument = NULL;

extern gmac::MemManager *memManager;

static void __attribute__((constructor)) gmacCudaInit(void)
{
	if((_cudaLaunch = (cudaLaunch_t)dlsym(RTLD_NEXT, "cudaLaunch")) == NULL)
		FATAL("cudaLaunch not found");
	if((_cudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT, "cudaConfigureCall")) == NULL)
		FATAL("cudaConfigureCall not found");
	if((_cudaSetupArgument = (cudaSetupArgument_t)dlsym(RTLD_NEXT, "cudaSetupArgument")) == NULL)
		FATAL("cudaSetupArgument not found");
}

cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
	size_t sharedMem, cudaStream_t stream)
{
	cudaError_t ret = cudaSuccess;
	TRACE("cudaConfigureCall");
	ret = _cudaConfigureCall(gridDim, blockDim, sharedMem, stream);
	return ret;
}

cudaError_t cudaSetupArgument(void *arg, size_t count, size_t offset)
{
	cudaError_t ret = cudaSuccess;
	TRACE("cudaSetupArgument");
	ret = _cudaSetupArgument(arg, count, offset);
	return ret;
}
