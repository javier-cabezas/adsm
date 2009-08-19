#include <config.h>
#include <threads.h>
#include <debug.h>

#include "GPUContext.h"

#include <string.h>
#include <assert.h>

#include <cuda.h>
#include <vector_types.h>
#include <driver_types.h>

#include <string>
#include <vector>

#define context \
	static_cast<gmac::GPUContext *>(PRIVATE_GET(gmac::Context::key))

#ifdef __cplusplus
extern "C" {
#endif

void **__cudaRegisterFatBinary(void *fatCubin)
{
	CUmodule *mod = new CUmodule;
	context->lock();
	CUresult ret = cuModuleLoadFatBinary(mod, fatCubin);
	context->release();
	assert(ret == CUDA_SUCCESS);
	return (void **)mod;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	context->lock();
	CUresult ret = cuModuleUnload(*mod);
	context->release();
	if(ret != CUDA_SUCCESS) return;
	delete mod;
}

void __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char *devFun,
		const char *devName, int threadLimit, uint3 *tid, uint3 *bid,
		dim3 *bDim, dim3 *gDim)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	CUfunction fun;
	context->lock();
	CUresult ret = cuModuleGetFunction(&fun, *mod, devName);
	context->release();
	if(ret != CUDA_SUCCESS) return;
	context->function(hostFun, fun);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int size,
		int constant, int global)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	CUdeviceptr ptr;
	unsigned int deviceSize;
	context->lock();
	CUresult ret = cuModuleGetGlobal(&ptr, &deviceSize, *mod, deviceName);
	context->release();
	if(ret != CUDA_SUCCESS) return;
	context->variable(hostVar, ptr, deviceSize);
}


void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
		size_t size, size_t alignment, int storage)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, int tokens)
{
	context->call(gridDim, blockDim, sharedMem, tokens);
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	context->argument(arg, count, offset);
	return cudaSuccess;
}

extern gmacError_t gmacLaunch(const char *);
cudaError_t cudaLaunch(const char *symbol)
{
	gmacLaunch(symbol);
	return cudaSuccess;
}

#ifdef __cplusplus
}
#endif


