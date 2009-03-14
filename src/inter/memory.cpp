#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <dlfcn.h>

#include <driver_types.h>

#define PARAVER_NO_CUDA_OVERRIDE

#include <common/config.h>
#include <common/debug.h>
#include <common/MemManager.h>

#define EXTERN extern "C"

static const size_t pageSize = 0x1000;

static cudaError_t (*_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*_cudaFree)(void *) = NULL;
static cudaError_t (*_cudaMallocPitch)(void **, size_t *, size_t, size_t) = NULL;
static cudaError_t (*_cudaLaunch)(const char *) = NULL;
cudaError_t (*_cudaThreadSynchronize)(void) = NULL;

static gmac::MemManager *memManager = NULL;

static struct timeval start, end;

static void __attribute__((constructor)) gmacInit(void)
{
	gettimeofday(&start, NULL);
	if((_cudaMalloc = (cudaError_t (*)(void **, size_t))dlsym(RTLD_NEXT, "cudaMalloc")) == NULL)
		FATAL("cudaMalloc not found");
	if((_cudaFree = (cudaError_t (*)(void *))dlsym(RTLD_NEXT, "cudaFree")) == NULL)
		FATAL("cudaFree not found");
	if((_cudaMallocPitch = (cudaError_t (*)(void **, size_t *, size_t, size_t))dlsym(RTLD_NEXT, "cudaMallocPitch")) == NULL)
		FATAL("cudaMallocPitch not found");
	if((_cudaLaunch = (cudaError_t (*)(const char *))dlsym(RTLD_NEXT, "cudaLaunch")) == NULL)
		FATAL("cudaLaunch not found");
	if((_cudaThreadSynchronize = (cudaError_t (*)(void))dlsym(RTLD_NEXT, "cudaThreadSynchronize")) == NULL)
		FATAL("cudaThreadSynchronize not found");

	memManager = gmac::getManager(getenv(memManagerVar));
}

static void __attribute__((destructor)) gmacFini(void)
{
	gettimeofday(&end, NULL);
	double sec = end.tv_sec - start.tv_sec;
	double musec = end.tv_usec - start.tv_usec;
	fprintf(stderr,"Time: %.3f\n", (sec * 10e6 + musec) / 10e6);
	if(memManager) delete memManager;
}

EXTERN cudaError_t cudaMalloc(void **devPtr, size_t count)
{
	cudaError_t ret = cudaSuccess;

	count = (count < pageSize) ? pageSize : count;

	/* Call CUDA library to get the memory */
	if((ret = _cudaMalloc(devPtr, count)) != cudaSuccess)
		return ret;

	if(!memManager) return ret;

	/* Notify the allocation to the memory manager */
	if(!memManager->alloc(*devPtr, count)) {
		_cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}

	return cudaSuccess;
}

EXTERN cudaError_t cudaFree(void *devPtr)
{
	_cudaFree(devPtr);
	if(!memManager) return cudaSuccess;
	memManager->release(devPtr);
	return cudaSuccess;
}


EXTERN cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
		size_t widthInBytes, size_t height)
{
	void *cpuAddr = NULL;
	cudaError_t ret = cudaSuccess;
	size_t count = widthInBytes * height;

	/* Make sure we at least request one page */
	if(count < pageSize) {
		height = pageSize / widthInBytes;
		if(pageSize % widthInBytes) height++;
	}

	/* Call the CUDA library */
	if((ret = _cudaMallocPitch(devPtr, pitch, widthInBytes,
			height)) != cudaSuccess)
		return ret;

	if(!memManager) return ret;

	/* Notify to the memory manager */
	if(!memManager->alloc(*devPtr, *pitch)) {
		_cudaFree(*devPtr);
		return cudaErrorMemoryAllocation;
	}

	return cudaSuccess;
}

EXTERN cudaError_t cudaLaunch(const char *symbol)
{
	if(memManager) memManager->execute();
	return _cudaLaunch(symbol);
}

EXTERN cudaError_t cudaThreadSynchronize(void)
{
	cudaError_t ret;
	ret = _cudaThreadSynchronize();
	if(memManager) memManager->sync();

	return ret;
}
