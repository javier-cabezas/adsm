#include <config.h>

#include <gmac/init.h>

#include <kernel/Kernel.h>

#include "Accelerator.h"
#include "Mode.h"
#include "Module.h"

#include <cstring>

#include <cuda.h>
#include <vector_types.h>
#include <driver_types.h>

#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

using gmac::cuda::Accelerator;
using gmac::cuda::Mode;
using gmac::KernelDescriptor;
using gmac::cuda::ModuleDescriptor;
using gmac::cuda::TextureDescriptor;
using gmac::cuda::VariableDescriptor;

/*!
 * @returns Module **
 */
void **__cudaRegisterFatBinary(void *fatCubin)
{
    gmac::util::Logger::TRACE("CUDA Fat binary: %p", fatCubin);
    gmac::util::Logger::ASSERTION(gmac::proc->nAccelerators() > 0);
    gmac::enterGmac();
    // Use the first GPU to load the fat binary
    void **ret = (void **) new ModuleDescriptor(fatCubin);
	gmac::exitGmac();
	return ret;
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
	gmac::enterGmac();
    ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
    delete mod;
	gmac::exitGmac();
}

void __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char *devFun,
		const char *devName, int threadLimit, uint3 *tid, uint3 *bid,
		dim3 *bDim, dim3 *gDim)
{
    gmac::util::Logger::TRACE("CUDA Function");
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	gmac::util::Logger::ASSERTION(mod != NULL);
	gmac::enterGmac();
    KernelDescriptor k = KernelDescriptor(devName, (gmacKernel_t) hostFun);
    mod->add(k);
	gmac::exitGmac();
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char *deviceAddress, const char *deviceName, int ext, int size,
		int constant, int global)
{
    gmac::util::Logger::TRACE("CUDA Variable %s", deviceName);
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	gmac::util::Logger::ASSERTION(mod != NULL);
	gmac::enterGmac();
    VariableDescriptor v = VariableDescriptor(deviceName, hostVar, bool(constant));
    mod->add(v);
	gmac::exitGmac();
}

void __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
		const void **deviceAddress, const char *deviceName, int dim, int norm, int ext)
{
    gmac::util::Logger::TRACE("CUDA Texture");
	ModuleDescriptor *mod = (ModuleDescriptor *)fatCubinHandle;
	gmac::util::Logger::ASSERTION(mod != NULL);
	gmac::enterGmac();
    TextureDescriptor t = TextureDescriptor(deviceName, hostVar);
	mod->add(t);
	gmac::exitGmac();
}

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr)
{
}

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr,
		size_t size, size_t alignment, int storage)
{
}

cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, cudaStream_t tokens)
{
	gmac::enterGmac();
    Mode *mode = dynamic_cast<Mode *>(gmac::Mode::current());
	mode->call(gridDim, blockDim, sharedMem, tokens);
	gmac::exitGmac();
	return cudaSuccess;
}

cudaError_t cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	gmac::enterGmac();
    Mode *mode = dynamic_cast<Mode *>(gmac::Mode::current());
	mode->argument(arg, count, offset);
	gmac::exitGmac();
	return cudaSuccess;
}

extern gmacError_t gmacLaunch(gmacKernel_t k);
cudaError_t cudaLaunch(gmacKernel_t k)
{
	gmacError_t ret = gmacLaunch(k);
    assert(ret == gmacSuccess);
	return cudaSuccess;
}

#ifdef __cplusplus
}
#endif


