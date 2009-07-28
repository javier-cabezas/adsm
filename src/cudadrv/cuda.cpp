#include "api.h"
#include "driver.h"

#include <config/debug.h>

#include <cuda.h>

#include <string>
#include <list>

static void __attribute__((constructor(102))) gmacCudaInit(void)
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
		case cudaErrorLaunchFailure:
			return "Failure during CUDA kernel launch";
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
		case CUDA_ERROR_LAUNCH_FAILED:
			return __gmacReturn(gmacErrorLaunchFailure);
		default: return __gmacReturn(gmacErrorUnknown);
	}
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

extern std::list<CUtexref *> __textures;
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
	// Set-up textures (if any)
	std::list<CUtexref *>::const_iterator t;
	for(t = __textures.begin(); t != __textures.end(); t++) {
		cuParamSetTexRef(fun, CU_PARAM_TR_DEFAULT, *(*t));
	}
	// Set shared size
	TRACE("Dynamic Shared %d", gmacCall.shared);
	if((ret = cuFuncSetSharedSize(fun, gmacCall.shared)) != CUDA_SUCCESS)
		return __gmacError(ret);
	// Set block shape
	TRACE("Block Size %d, %d, %d", gmacCall.block.x, gmacCall.block.y,
			gmacCall.block.z);
	if((ret = cuFuncSetBlockShape(fun,
			gmacCall.block.x, gmacCall.block.y, gmacCall.block.z)
		) != CUDA_SUCCESS) return __gmacError(ret);
	// Call the kernel
	TRACE("Grid Size %d, %d, %d", gmacCall.grid.x, gmacCall.grid.y);
	if((ret = cuLaunchGrid(fun,
			gmacCall.grid.x, gmacCall.grid.y)) != CUDA_SUCCESS)
		return __gmacError(ret);
	TRACE("Kernel launched");
	return __gmacError(ret);
}

gmacError_t __gmacThreadSynchronize()
{
	TRACE("Kernel Synchronize");
	CUresult ret = cuCtxSynchronize();
	return __gmacError(ret);
}
