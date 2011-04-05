#include "config/common.h"
#include "include/gmac/opencl.h"

#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/hpe/Mode.h"

#if defined(GMAC_DLL)
#include "gmac/init.h"
#endif

using __impl::opencl::hpe::Accelerator;
using __impl::opencl::hpe::Mode;

GMAC_API gmacError_t APICALL __oclSetArgument(const void *addr, size_t size, unsigned index)
{
    gmac::enterGmac();
    Mode &mode = Mode::getCurrent();
    gmacError_t ret = mode.argument(addr, size, index);
    gmac::exitGmac();

    return ret;
}

GMAC_API gmacError_t APICALL __oclConfigureCall(size_t workDim, size_t *globalWorkOffset,
        size_t *globalWorkSize, size_t *localWorkSize)
{
    gmac::enterGmac();
    Mode &mode = Mode::getCurrent();
    gmacError_t ret = mode.call(cl_int(workDim), globalWorkOffset, 
        globalWorkSize, localWorkSize);
    gmac::exitGmac();
    return ret;
}

GMAC_API gmacError_t APICALL __oclLaunch(gmacKernel_t k)
{
    return gmacLaunch(k);
}

GMAC_API gmacError_t APICALL __oclPrepareCLCode(const char *code, const char *flags)
{
    gmac::enterGmac();
    gmacError_t ret = Accelerator::prepareCLCode(code, flags);
    gmac::exitGmac();

    return ret;
}

gmacError_t APICALL __oclPrepareCLBinary(const unsigned char *binary, size_t size, const char *flags)
{
    gmac::enterGmac();
    gmacError_t ret = Accelerator::prepareCLBinary(binary, size, flags);
    gmac::exitGmac();

    return ret;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
