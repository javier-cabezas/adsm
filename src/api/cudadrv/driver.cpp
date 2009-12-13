#include <config.h>
#include <debug.h>

#include <gmac/init.h>

#include "Context.h"

#include <cstring>
#include <cassert>

#include <cuda.h>
#include <vector_types.h>
#include <driver_types.h>

#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

using gmac::gpu::Context;
using gmac::gpu::Module;

void **__cudaRegisterFatBinary(void *fatCubin)
{
	__enterGmac();
	void **ret = (void **)Context::current()->load(fatCubin);
	__exitGmac();
	return ret;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	__enterGmac();
	Module *mod = (Module *)fatCubinHandle;
	Context::current()->unload(mod);
	__exitGmac();
}

void __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char *devFun,
		const char *devName, int threadLimit, uint3 *tid, uint3 *bid,
		dim3 *bDim, dim3 *gDim)
{
	Module *mod = (Module *)fatCubinHandle;
	assert(mod != NULL);
	__enterGmac();
	Context::current()->lock();
	mod->function(hostFun, devName);
	Context::current()->unlock();
	__exitGmac();
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int size,
		int constant, int global)
{
	Module *mod = (Module *)fatCubinHandle;
	assert(mod != NULL);
	__enterGmac();
	Context::current()->lock();
	if(constant == 0) mod->variable(hostVar, deviceName);
	else mod->constant(hostVar, deviceName);
	mod->variable(hostVar, deviceName);
	Context::current()->unlock();
	__exitGmac();
}


void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr)
{
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
		size_t size, size_t alignment, int storage)
{
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, int tokens)
{
	__enterGmac();
	Context::current()->call(gridDim, blockDim, sharedMem, tokens);
	__exitGmac();
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	__enterGmac();
	Context::current()->argument(arg, count, offset);
	__exitGmac();
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


