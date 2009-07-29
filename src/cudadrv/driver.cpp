#include "api.h"
#include "driver.h"

#include <config/config.h>
#include <config/debug.h>

#include <cuda.h>
#include <string.h>
#include <assert.h>

#include <string>
#include <vector>

gmacError_t gmacLastError = gmacSuccess;

FunctionMap funMap;
VariableMap varMap;
std::vector<gmacCall_t> gmacCallStack;
size_t gmacStackPtr = 0;
uint8_t gmacStack[gmacStackSize];
size_t gmacGlobalShared = 0;

#ifdef __cplusplus
extern "C" {
#endif

void **__cudaRegisterFatBinary(void *fatCubin)
{
	CUmodule *mod = new CUmodule;
	CUresult ret = cuModuleLoadFatBinary(mod, fatCubin);
	__gmacError(ret);
	if(ret != CUDA_SUCCESS) return NULL;
	return (void **)mod;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	CUresult ret = cuModuleUnload(*mod);
	__gmacError(ret);
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
	CUresult ret = cuModuleGetFunction(&fun, *mod, devName);
	__gmacError(ret);
	if(ret != CUDA_SUCCESS) return;
	funMap.insert(FunctionMap::value_type(hostFun, fun));
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int size,
		int constant, int global)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	CUdeviceptr ptr;
	unsigned int deviceSize;
	CUresult ret = cuModuleGetGlobal(&ptr, &deviceSize, *mod, deviceName);
	__gmacError(ret);
	if(ret != CUDA_SUCCESS) return;
	struct __deviceVariable variable;
	variable.ptr = ptr;
	variable.size = deviceSize;
	variable.constant = (constant != 0) ? true : false;
	varMap.insert(VariableMap::value_type(hostVar, variable));
	TRACE("CUDA Variable %s @ 0x%x (%d bytes)", deviceName, ptr, deviceSize);
}


void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	TRACE("RegisterVar %p", devicePtr);
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
		size_t size, size_t alignment, int storage)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	TRACE("RegisterSharedVar %p (%d bytes) aligned to %d in %d",
			devicePtr, size, alignment, storage);
//	int rem = gmacGlobalShared % alignment;
//	if(rem != 0) gmacGlobalShared += (alignment - rem);
//	gmacGlobalShared += size;
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, int tokens)
{
	gmacCall_t gmacCall;
	gmacCall.grid = gridDim;
	gmacCall.block = blockDim;
	gmacCall.shared = sharedMem;
	gmacCall.tokens = tokens;
	gmacCall.stack = gmacStackPtr;
	gmacCallStack.push_back(gmacCall);
	return __gmacReturn(gmacSuccess);
}

cudaError_t cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	TRACE("cudaSetupArgument @ %d (%d bytes)", offset, count);
	memcpy(&gmacStack[offset], arg, count);
	size_t top = offset + count;
	gmacStackPtr = (gmacStackPtr > top) ? gmacStackPtr : top;
	return __gmacReturn(gmacSuccess);
}

extern gmacError_t gmacLaunch(const char *);
cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

#ifdef __cplusplus
}
#endif


