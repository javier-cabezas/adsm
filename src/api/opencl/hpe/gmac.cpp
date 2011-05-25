#include <fstream>

#include "config/common.h"
#include "include/gmac/opencl.h"

#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/hpe/Mode.h"
#include "api/opencl/hpe/Kernel.h"
#include "memory/Manager.h"

#include "hpe/init.h"

using __impl::util::params::ParamAutoSync;

static inline __impl::opencl::hpe::Mode &getCurrentOpenCLMode()
{
	return dynamic_cast<__impl::opencl::hpe::Mode &>(__impl::core::hpe::getCurrentMode());
}

GMAC_API gmacError_t APICALL oclKernelSetArg(ocl_kernel kernel, unsigned index, const void *addr, size_t size)
{
    enterGmac();
    ((__impl::opencl::hpe::KernelLaunch *)kernel.impl_)->setArgument(addr, size, index);
    exitGmac();

    return gmacSuccess;
}

gmacError_t gmacLaunch(__impl::core::hpe::KernelLaunch &);
gmacError_t gmacThreadSynchronize(__impl::core::hpe::KernelLaunch &);

GMAC_API gmacError_t APICALL oclKernelLaunch(ocl_kernel kernel,
    size_t workDim, size_t *globalWorkOffset,
    size_t *globalWorkSize, size_t *localWorkSize)
{
    enterGmac();

    __impl::opencl::hpe::KernelLaunch *launch = (__impl::opencl::hpe::KernelLaunch *)kernel.impl_;

    launch->setConfiguration(cl_int(workDim), globalWorkOffset, globalWorkSize, localWorkSize);
    gmacError_t ret = gmacLaunch(*launch);
    if(ret == gmacSuccess) {
#if defined(SEPARATE_COMMAND_QUEUES)
        ret = gmacThreadSynchronize(*(__impl::opencl::hpe::KernelLaunch *)kernel.impl_);
#else
        ret = __impl::memory::getManager().acquireObjects(getCurrentOpenCLMode());
#endif
    }
    exitGmac();
    return ret;
}


#if 0
GMAC_API gmacError_t APICALL __oclKernelWait(ocl_kernel *kernel)
{
    enterGmac();
    gmacError_t ret = gmacThreadSynchronize(*(__impl::opencl::hpe::KernelLaunch *)kernel->impl_);
    exitGmac();
    return ret;
}
#endif


GMAC_API gmacError_t APICALL oclCompileSource(const char *code, const char *flags)
{
    enterGmac();
    gmacError_t ret = __impl::opencl::hpe::Accelerator::prepareCLCode(code, flags);
    exitGmac();

    return ret;
}

GMAC_API gmacError_t APICALL oclCompileSourceFile(const char *path, const char *flags)
{
    std::ifstream in(path, std::ios_base::in);
    if (!in.good()) return gmacErrorInvalidValue;
    in.seekg (0, std::ios::end);
    std::streampos length = in.tellg();
    in.seekg (0, std::ios::beg);
	if (length == std::streampos(0)) return gmacSuccess;
    // Allocate memory for the code
    char *buffer = new char[int(length)+1];
    // Read data as a block
    in.read(buffer,length);
    buffer[length] = '\0';
    in.close();
    gmacError_t ret = oclCompileSource(buffer, flags);
    in.close();
    delete [] buffer;

    return ret;
}


gmacError_t APICALL oclCompileBinary(const unsigned char *binary, size_t size, const char *flags)
{
    enterGmac();
    gmacError_t ret = __impl::opencl::hpe::Accelerator::prepareCLBinary(binary, size, flags);
    exitGmac();

    return ret;
}

gmacError_t APICALL oclCompileBinaryFile(const char *path, const char *flags)
{
    std::ifstream in(path, std::ios_base::in);
    if (!in.good()) return gmacErrorInvalidValue;
    in.seekg (0, std::ios::end);
    std::streampos length = in.tellg();
    in.seekg (0, std::ios::beg);
	if (length == std::streampos(0)) return gmacSuccess;
    // Allocate memory for the code
    unsigned char *buffer = new unsigned char[int(length)+1];
    // Read data as a block
    in.read((char *) buffer,length);
    buffer[length] = '\0';
    in.close();
    gmacError_t ret = oclCompileBinary(buffer, length, flags);
    in.close();
    delete [] buffer;

    return ret;
}

gmacError_t APICALL oclKernelGet(gmac_kernel_id_t id, ocl_kernel *kernel)
{
    enterGmac();
    __impl::core::hpe::Mode &mode = getCurrentOpenCLMode();
    __impl::core::hpe::KernelLaunch *launch;
    gmacError_t ret = mode.launch(id, launch);
    if (ret == gmacSuccess) {
        kernel->impl_ = launch;
    }
    exitGmac();

    return gmacSuccess;
}

gmacError_t APICALL oclKernelDestroy(ocl_kernel kernel)
{
    enterGmac();
    if (kernel.impl_ != NULL) {
        delete ((__impl::opencl::hpe::KernelLaunch *) kernel.impl_);
    }
    exitGmac();

    return gmacSuccess;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
