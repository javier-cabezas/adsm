#include "api.h"
#include "driver.h"

#include <config/debug.h>

#include <cuda.h>

#include <string>

static void __attribute__((constructor(101))) gmacCudaInit(void)
{
	if(cuInit(0) != CUDA_SUCCESS)
		FATAL("Unable to init CUDA");
	// TODO: store device/context management
	int devCount = 0;
	if(cuDeviceGetCount(&devCount) != CUDA_SUCCESS || devCount == 0)
		FATAL("No CUDA-enable devices found");
	CUdevice cuDev;
	if(cuDeviceGet(&cuDev, 0) != CUDA_SUCCESS)
		FATAL("Unable to access CUDA device");
	CUcontext cuCtx;
	if((cuCtxCreate(&cuCtx, 0, cuDev) != CUDA_SUCCESS))
		FATAL("Unable to create CUDA context");
}

#ifdef __cplusplus
extern "C" {
#endif

gmacError_t gmacGetLastError() { return gmacLastError; }

const char *gmacGetErrorString(gmacError_t error)
{
	switch(error) {
		case cudaErrorStartupFailure:
			return "Failure during CUDA initialization";
		case cudaErrorApiFailureBase:
			return "Failure during CUDA API call";
		default: return NULL;
	}
}

#ifdef __cplusplus
}
#endif

gmacError_t __gmacError(CUresult ret)
{
	switch(ret) {
		case CUDA_SUCCESS: return __gmacReturn(gmacSuccess);
		case CUDA_ERROR_OUT_OF_MEMORY:
			return __gmacReturn(gmacErrorMemoryAllocation);
		default: return __gmacReturn(gmacErrorUnknown);
	}
}

static inline CUdeviceptr voidToDev(void *v)
{
	unsigned long u = (unsigned long)v;
	return (CUdeviceptr)(u & 0xffffffff);
}

static inline CUdeviceptr voidToDev(const void *v)
{
	unsigned long u = (unsigned long)v;
	return (CUdeviceptr)(u & 0xffffffff);
}

gmacError_t __gmacMalloc(void **devPtr, size_t count)
{
	*devPtr = NULL;
	CUresult ret = cuMemAlloc((CUdeviceptr *)devPtr, count);
	return __gmacError(ret);
}

gmacError_t __gmacMallocPitch(void **devPtr, size_t *pitch,
		size_t widthInBytes, size_t height)
{
	CUresult ret = cuMemAllocPitch((CUdeviceptr *)devPtr, (unsigned *)pitch,
		widthInBytes, height, sizeof(float));
	return __gmacError(ret);
}

gmacError_t __gmacFree(void *devPtr)
{
	CUresult ret = cuMemFree(voidToDev(devPtr));
	return __gmacError(ret);
}

gmacError_t __gmacMemcpyToDevice(void *devPtr, const void *cpuPtr, size_t n)
{
	CUresult ret = cuMemcpyHtoD(voidToDev(devPtr), cpuPtr, n);
	return __gmacError(ret);
}

gmacError_t __gmacMemcpyToHost(void *cpuPtr, const void *devPtr, size_t n)
{
	CUresult ret = cuMemcpyDtoH(cpuPtr, voidToDev(devPtr), n);
	return __gmacError(ret);
}

gmacError_t __gmacMemcpyDevice(void *dstPtr, const void *srcPtr, size_t n)
{
	CUresult ret = cuMemcpyDtoD(voidToDev(dstPtr), voidToDev(srcPtr), n);
	return __gmacError(ret);
}

gmacError_t __gmacMemcpyToDeviceAsync(void *devPtr, const void *cpuPtr,
		size_t n)
{
	CUresult ret = cuMemcpyHtoDAsync(voidToDev(devPtr), cpuPtr, n, 0);
	return __gmacError(ret);
}

gmacError_t __gmacMemcpyToHostAsync(void *cpuPtr, const void *devPtr,
		size_t n)
{
	CUresult ret = cuMemcpyDtoHAsync(cpuPtr, voidToDev(devPtr), n, 0);
	return __gmacError(ret);
}

gmacError_t __gmacMemset(void *devPtr, int i, size_t n)
{
	CUresult ret = CUDA_SUCCESS;
	unsigned char c = i & 0xff;
	if((n % 4) == 0) {
		unsigned m = c | (c << 8);
		m |= (m << 16);
		ret = cuMemsetD32(voidToDev(devPtr), m, n / 4);
	}
	else if((n % 2) == 0) {
		unsigned short s = c | (c << 8);
		ret = cuMemsetD16(voidToDev(devPtr), s, n / 2);
	}
	else {
		ret = cuMemsetD8(voidToDev(devPtr), c, n);
	}
	return __gmacError(ret);
}

gmacError_t __gmacLaunch(const char *symbol)
{
	gmacCall_t gmacCall = gmacCallStack.back();
	gmacCallStack.pop_back();
	size_t count = gmacStackPtr - gmacCall.stack;
	gmacStackPtr = gmacCall.stack;

	// Get the device function
	HASH_MAP<std::string, CUfunction>::const_iterator i;
	if((i = funMap.find(symbol)) == funMap.end())
		return __gmacReturn(gmacErrorInvalidDeviceFunction);
	CUfunction fun = i->second;
	// Copy parameters
	CUresult ret = cuParamSetv(fun, 0, &gmacStack[gmacCall.stack], count);
	if(ret != CUDA_SUCCESS) return __gmacError(ret);
	if((ret = cuParamSetSize(fun, count)) != CUDA_SUCCESS)
		return __gmacError(ret);
	// Set block shape
	TRACE("Block: (%d, %d, %d)",
			gmacCall.block.x, gmacCall.block.y, gmacCall.block.z);
	if((ret = cuFuncSetBlockShape(fun,
			gmacCall.block.x, gmacCall.block.y, gmacCall.block.z)
		) != CUDA_SUCCESS) return __gmacError(ret);
	// Set shared size
	if(gmacCall.shared &&
			((ret = cuFuncSetSharedSize(fun, gmacCall.shared)) != CUDA_SUCCESS))
		return __gmacError(ret);
	// Call the kernel
	TRACE("Grid: (%d, %d)", gmacCall.grid.x, gmacCall.grid.y);
	if((ret = cuLaunchGrid(fun,
			gmacCall.grid.x, gmacCall.grid.y)) != CUDA_SUCCESS)
		return __gmacError(ret);
}

gmacError_t __gmacThreadSynchronize()
{
	CUresult ret = cuCtxSynchronize();
	return __gmacError(ret);
}
