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

#include <cuda_runtime.h>

static HASH_MAP<pid_t, gmac::MemManager *> *gmacMemManagers = NULL;
static size_t pageSize = 0;

#define memManager (*gmacMemManagers)[gettid()]

void gmacRemoveManager(void)
{
	if(gmacMemManagers->find(gettid()) != gmacMemManagers->end()) {
		delete memManager;
		gmacMemManagers->erase(gettid());
	}
}

void gmacCreateManager(void)
{
	memManager = gmac::getManager(getenv(memManagerVar));
}

static void __attribute__((constructor)) gmacInit(void)
{
	pageSize = getpagesize();
	gmacMemManagers = new HASH_MAP<pid_t, gmac::MemManager *>();
	gmacCreateManager();
}

static void __attribute__((destructor)) gmacFini(void)
{
	gmacRemoveManager();
	assert(gmacMemManagers->empty());
	delete gmacMemManagers;
}

cudaError_t gmacMalloc(void **devPtr, size_t count)
{
	cudaError_t ret = cudaSuccess;
	count = (count < pageSize) ? pageSize : count;
	if((ret = cudaMalloc(devPtr, count)) != cudaSuccess) {
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
	if((ret = cudaMalloc(&devPtr, count)) != cudaSuccess) {
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
	if(memManager) memManager->release(devPtr);
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

extern cudaError_t (*_cudaLaunch)(const char *);
cudaError_t gmacLaunch(const char *symbol)
{
	TRACE("gmacLaunch");
	if(memManager) memManager->execute();
	return _cudaLaunch(symbol);
}

cudaError_t gmacThreadSynchronize()
{
	TRACE("gmacThreadSynchronize");
	cudaError_t ret = cudaThreadSynchronize();
	if(memManager) memManager->sync();
	return ret;
}

cudaError_t gmacSetupArgument(void *arg, size_t size, size_t offset)
{
	TRACE("gmacSetupArgument");
	return cudaSetupArgument(arg, size, offset);
}
