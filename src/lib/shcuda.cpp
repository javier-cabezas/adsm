#ifdef NATIVE
#undef NATIVE
#endif

#include "shcuda.h"

#include <stdlib.h>
#include <unistd.h>
#include <common/config.h>
#include <common/debug.h>
#include <common/MemManager.h>

#include <cuda_runtime.h>

static icuda::MemManager *memManager = NULL;
static size_t pageSize = 0;

static void __attribute__((constructor)) shCudaInit(void)
{
	pageSize = getpagesize();
	memManager = icuda::getManager(getenv(memManagerVar));
}

static void __attribute__((destructor)) shCudaFini(void)
{
	if(memManager) delete memManager;
}

cudaError_t shCudaMalloc(void **devPtr, size_t count)
{
	cudaError_t ret = cudaSuccess;
	count = (count < pageSize) ? pageSize : count;
	if((ret = cudaMalloc(devPtr, count)) != cudaSuccess)
		return ret;
	if(!memManager) return ret;
	if(!memManager->alloc(*devPtr, count)) {
		cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}
	return cudaSuccess;
}

cudaError_t shCudaFree(void *devPtr)
{
	cudaFree(devPtr);
	if(memManager) memManager->release(devPtr);
	return cudaSuccess;
}

cudaError_t shCudaMallocPitch(void **devPtr, size_t *pitch,
		size_t widthInBytes, size_t height)
{
	void *cpuAddr = NULL;
	cudaError_t ret = cudaSuccess;
	size_t count = widthInBytes * height;

	if(count < pageSize) {
		height = pageSize / widthInBytes;
		if(pageSize % widthInBytes) height++;
	}

	if((ret = cudaMallocPitch(devPtr, pitch, widthInBytes,
			height)) != cudaSuccess)
		return ret;

	if(!memManager) return ret;

	if(!memManager->alloc(*devPtr, *pitch)) {
		cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}

	return cudaSuccess;
}

cudaError_t shCudaLaunch(const char *symbol)
{
	if(memManager) memManager->execute();
	return cudaLaunch(symbol);
}

cudaError_t shCudaThreadSynchronize()
{
	cudaError_t ret = cudaThreadSynchronize();
	if(memManager) memManager->sync();
	return ret;
}
