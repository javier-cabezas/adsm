#include "api.h"
#include "driver.h"

#include <config/config.h>

#include <cuda.h>
#include <string.h>
#include <assert.h>

#include <string>
#include <vector>

gmacError_t gmacLastError = gmacSuccess;

HASH_MAP<std::string, CUfunction> funMap;
std::vector<gmacCall_t> gmacCallStack;
size_t gmacStackPtr = 0;
uint8_t gmacStack[gmacStackSize];

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
	funMap[hostFun] = fun;
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


