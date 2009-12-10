#include <config.h>
#include <debug.h>

#include "Context.h"

#include <string.h>
#include <assert.h>

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
	return (void **)Context::current()->load(fatCubin);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	Module *mod = (Module *)fatCubinHandle;
	Context::current()->unload(mod);
}

void __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char *devFun,
		const char *devName, int threadLimit, uint3 *tid, uint3 *bid,
		dim3 *bDim, dim3 *gDim)
{
	Module *mod = (Module *)fatCubinHandle;
	assert(mod != NULL);
	Context::current()->lock();
	mod->function(hostFun, devName);
	Context::current()->unlock();
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int size,
		int constant, int global)
{
	Module *mod = (Module *)fatCubinHandle;
	assert(mod != NULL);
	Context::current()->lock();
	if(constant == 0) mod->variable(hostVar, deviceName);
	else mod->constant(hostVar, deviceName);
	mod->variable(hostVar, deviceName);
	Context::current()->unlock();
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
	Context::current()->call(gridDim, blockDim, sharedMem, tokens);
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	Context::current()->argument(arg, count, offset);
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


