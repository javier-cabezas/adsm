#include <cstring>

#include <vector_types.h>
#include <driver_types.h>

#include <string>
#include <vector>

#include "include/gmac/cuda.h"
#include "hpe/init.h"

#include "hal/types.h"
#include "core/hpe/thread.h"

#include "hal/cuda/module.h"


#ifdef __cplusplus
extern "C" {
#endif

using __impl::hal::cuda::kernel_descriptor;

using __impl::hal::cuda::module_descriptor;
using __impl::hal::cuda::kernel_t;

using __impl::core::hpe::kernel;
using __impl::core::hpe::process;
using __impl::core::hpe::resource_manager;
using __impl::core::hpe::thread;
using __impl::core::hpe::vdevice;

using __impl::core::hpe::getProcess;

using __impl::hal::cuda::module_descriptor;
using __impl::hal::cuda::texture_descriptor;
using __impl::hal::cuda::variable_descriptor;

static inline vdevice &get_current_virtual_device()
{
    return __impl::core::hpe::thread::get_current_thread().get_current_virtual_device();
}

/*!
 * @returns Module **
 */
GMAC_API void ** APICALL __cudaRegisterFatBinary(void *fatCubin)
{
    TRACE(GLOBAL, "CUDA Fat binary: %p", fatCubin);
    enterGmac();
    // Use the first GPU to load the fat binary
    void **ret = (void **) new module_descriptor(fatCubin);
	exitGmac();
	return ret;
}

GMAC_API void APICALL __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    module_descriptor *mod = (module_descriptor *)fatCubinHandle;
    delete mod;
}

GMAC_API void APICALL __cudaRegisterFunction(
		void **fatCubinHandle, const char *hostFun, char * /*devFun*/,
		const char *devName, int /*threadLimit*/, uint3 * /*tid*/, uint3 * /*bid*/,
		dim3 * /*bDim*/, dim3 * /*gDim*/)
{
    TRACE(GLOBAL, "CUDA Function");
	module_descriptor *mod = (module_descriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    kernel_descriptor k = kernel_descriptor(devName, (gmac_kernel_id_t) hostFun);
    mod->add(k);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
		char * deviceAddress, const char *deviceName, int /*ext*/, int /*size*/,
		int constant, int /*global*/)
{
    TRACE(GLOBAL, "CUDA Variable %s: %p", deviceName, deviceAddress);
	module_descriptor *mod = (module_descriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    variable_descriptor v = variable_descriptor(deviceName, hostVar, bool(constant != 0));
    mod->add(v);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
		const void ** /*deviceAddress*/, const char *deviceName, int /*dim*/, int /*norm*/, int /*ext*/)
{
    TRACE(GLOBAL, "CUDA Texture");
	module_descriptor *mod = (module_descriptor *)fatCubinHandle;
	ASSERTION(mod != NULL);
	enterGmac();
    texture_descriptor t = texture_descriptor(deviceName, hostVar);
	mod->add(t);
	exitGmac();
}

GMAC_API void APICALL __cudaRegisterShared(void ** /*fatCubinHandle*/, void ** /*devicePtr*/)
{
}

GMAC_API void APICALL __cudaRegisterSharedVar(void ** /*fatCubinHandle*/, void ** /*devicePtr*/,
		size_t /*size*/, size_t /*alignment*/, int /*storage*/)
{
}

GMAC_API cudaError_t APICALL cudaConfigureCall(dim3 gridDim, dim3 blockDim,
		size_t sharedMem, cudaStream_t tokens)
{
	enterGmac();
    kernel_t::config *config = new kernel_t::config(gridDim, blockDim, sharedMem, tokens);
    kernel::arg_list *args = new kernel::arg_list;
    thread::get_current_thread().new_kernel_launch(config, args);
	exitGmac();
	return cudaSuccess;
}

GMAC_API cudaError_t APICALL
cudaSetupArgument(const void *arg, size_t count, size_t offset)
{
	enterGmac();
    thread::pair_launch &launch = thread::get_current_thread().get_kernel_launch();
    launch.second->push_arg(arg, count);
	//mode.argument(arg, count, (off_t)offset);
	exitGmac();
	return cudaSuccess;
}

GMAC_API gmacError_t APICALL
gmacLaunch(const char *k, __impl::hal::kernel_t::config &config, kernel::arg_list &args);
                                                    
GMAC_API cudaError_t APICALL
cudaLaunch(const char *k)
{
    thread::pair_launch &conf = thread::get_current_thread().get_kernel_launch();

	gmacError_t ret = gmacLaunch(k, *conf.first, *conf.second);
    assert(ret == gmacSuccess);
	return cudaSuccess;
}

#ifdef __cplusplus
}
#endif


