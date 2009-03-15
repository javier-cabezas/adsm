#include <gmac.h>

#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <assert.h>

#include <common/config.h>
#include <common/threads.h>
#include <common/debug.h>
#include <common/paraver.h>
#include <common/MemManager.h>

#include <cuda_runtime.h>

MUTEX(gmacMutex);
gmac::MemManager *memManager = NULL;
static unsigned memManagerCount = 0;
static size_t pageSize = 0;

void gmacRemoveManager(void)
{
	__MUTEX_LOCK(gmacMutex);
	memManagerCount--;
	if(memManagerCount <= 0) {
		delete memManager;
		memManager = NULL;
	}
	__MUTEX_UNLOCK(gmacMutex);
}

void gmacCreateManager(void)
{
	__MUTEX_LOCK(gmacMutex);
	if(memManager == NULL)
		memManager = gmac::getManager(getenv(memManagerVar));
	memManagerCount++;
	__MUTEX_UNLOCK(gmacMutex);
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
	pushState(_gmacMalloc_);
	cudaError_t ret = cudaSuccess;
	count = (count < pageSize) ? pageSize : count;
	ret = cudaMalloc(devPtr, count);
	if(ret != cudaSuccess || !memManager) {
		popState();
		return ret;
	}
	if(!memManager->alloc(*devPtr, count)) {
		cudaFree(*devPtr);
		popState();
		return cudaErrorMemoryAllocation;
	}
	popState();
	return cudaSuccess;
}

cudaError_t gmacSafeMalloc(void **cpuPtr, size_t count)
{
	pushState(_gmacMalloc_);
	cudaError_t ret = cudaSuccess;
	void *devPtr;
	count = (count < pageSize) ? pageSize : count;
	ret = cudaMalloc(&devPtr, count);
	if(ret != cudaSuccess || !memManager) {
		popState();
		return ret;
	}
	if((*cpuPtr = memManager->safeAlloc(devPtr, count)) == NULL) {
		cudaFree(devPtr);
		popState();
		return cudaErrorMemoryAllocation;
	}
	popState();
	return cudaSuccess;
}

void *gmacSafePointer(void *devPtr)
{
	if(!memManager) return devPtr;
	return memManager->safe(devPtr);
}

cudaError_t gmacFree(void *devPtr)
{
	pushState(_gmacFree_);
	cudaFree(devPtr);
	if(memManager) {
		memManager->release(devPtr);
	}
	popState();
	return cudaSuccess;
}

cudaError_t gmacMallocPitch(void **devPtr, size_t *pitch,
		size_t widthInBytes, size_t height)
{
	pushState(_gmacMalloc_);
	void *cpuAddr = NULL;
	cudaError_t ret = cudaSuccess;
	size_t count = widthInBytes * height;

	if(count < pageSize) {
		height = pageSize / widthInBytes;
		if(pageSize % widthInBytes) height++;
	}

	ret = cudaMallocPitch(devPtr, pitch, widthInBytes, height);
	if(ret != cudaSuccess && !memManager) {
		popState();
		return ret;
	}

	if(!memManager->alloc(*devPtr, *pitch)) {
		cudaFree(*devPtr);
		popState();
		return cudaErrorMemoryAllocation;
	}

	popState();
	return cudaSuccess;
}

extern cudaError_t (*_cudaLaunch)(const char *);
cudaError_t gmacLaunch(const char *symbol)
{
	pushState(_gmacLaunch_);
	cudaError_t ret = cudaSuccess;
	if(memManager) {
		memManager->execute();
	}
	ret = _cudaLaunch(symbol);
	popState();
	return ret;
}

cudaError_t gmacThreadSynchronize()
{
	pushState(_gmacSync_);
	cudaError_t ret = cudaThreadSynchronize();
	if(memManager) {
		memManager->sync();
	}
	popState();
	return ret;
}
