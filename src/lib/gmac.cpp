#define NATIVE
#include "gmac.h"

#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>

#include <common/config.h>
#include <common/threads.h>
#include <common/debug.h>
#include <common/MemManager.h>

#ifdef PARAVER
#include "paraver.h"
#endif

#include <cuda_runtime.h>

MUTEX(gmacMutex);
gmac::MemManager *memManager = NULL;
static unsigned memManagerCount = 0;
static size_t pageSize = 0;

void gmacRemoveManager(void)
{
	MUTEX_LOCK(gmacMutex);
	memManagerCount--;
	if(memManagerCount <= 0) {
		delete memManager;
		memManager = NULL;
	}
	MUTEX_UNLOCK(gmacMutex);
}

void gmacCreateManager(void)
{
	MUTEX_LOCK(gmacMutex);
	if(memManager == NULL)
		memManager = gmac::getManager(getenv(memManagerVar));
	memManagerCount++;
	MUTEX_UNLOCK(gmacMutex);
}

static void __attribute__((constructor)) gmacInit(void)
{
	pageSize = getpagesize();
	MUTEX_INIT(gmacMutex);
	gmacCreateManager();
}

static void __attribute__((destructor)) gmacFini(void)
{
	gmacRemoveManager();
}

cudaError_t gmacMalloc(void **devPtr, size_t count)
{
	cudaError_t ret = cudaSuccess;
	count = (count < pageSize) ? pageSize : count;
	ret = cudaMalloc(devPtr, count);
	if(ret != cudaSuccess) {
		return ret;
	}
	if(!memManager) return ret;
	if(!memManager->alloc(*devPtr, count)) {
		cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}
	return cudaSuccess;
}

cudaError_t gmacSafeMalloc(void **cpuPtr, size_t count)
{
	cudaError_t ret = cudaSuccess;
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	ret = cudaMalloc(&devPtr, count);
	if(ret != cudaSuccess) {
		return ret;
	}
	if(!memManager) return ret;
	if((*cpuPtr = memManager->safeAlloc(devPtr, count)) == NULL) {
		cudaFree(devPtr);
		return cudaErrorMemoryAllocation;
	}
	return cudaSuccess;

}

void *gmacSafePointer(void *devPtr)
{
	if(!memManager) return devPtr;
	return memManager->safe(devPtr);
}

cudaError_t gmacFree(void *devPtr)
{
	cudaFree(devPtr);
	if(memManager) {
		memManager->release(devPtr);
	}
	return cudaSuccess;
}

cudaError_t gmacMallocPitch(void **devPtr, size_t *pitch,
		size_t widthInBytes, size_t height)
{
	void *cpuAddr = NULL;
	cudaError_t ret = cudaSuccess;
	size_t count = widthInBytes * height;

	if(count < pageSize) {
		height = pageSize / widthInBytes;
		if(pageSize % widthInBytes) height++;
	}

	ret = cudaMallocPitch(devPtr, pitch, widthInBytes, height);
	if(ret != cudaSuccess) return ret;

	if(!memManager) return ret;

	if(!memManager->alloc(*devPtr, *pitch)) {
		cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}

	return cudaSuccess;
}

extern cudaError_t (*_cudaLaunch)(const char *);
cudaError_t gmacLaunch(const char *symbol)
{
	cudaError_t ret = cudaSuccess;
	TRACE("gmacLaunch");
	if(memManager) {
		memManager->execute();
	}
	ret = _cudaLaunch(symbol);
	return ret;
}

cudaError_t gmacThreadSynchronize()
{
	TRACE("gmacThreadSynchronize");
	cudaError_t ret = cudaThreadSynchronize();
	if(memManager) {
		memManager->sync();
	}
	return ret;
}

cudaError_t gmacSetupArgument(void *arg, size_t size, size_t offset)
{
	cudaError_t ret;
	TRACE("gmacSetupArgument");
	ret = cudaSetupArgument(arg, size, offset);
	return ret;
}
