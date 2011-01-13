#include "config/common.h"

#include "api/opencl/Mode.h"

using __impl::opencl::Mode;

gmacError_t APICALL __oclPushArgumentWithSize(void *addr, size_t size)
{
    gmac::enterGmac();
    Mode &mode = Mode::current();
    gmacError_t ret = mode.argument(addr, size);
    gmac::exitGmac();

    return ret;
}

gmacError_t GMAC_LOCAL gmacLaunch(gmacKernel_t k);

GMAC_API gmacError_t APICALL __oclLaunch(gmacKernel_t k)
{
    return gmacLaunch(k);
}

gmacError_t APICALL __oclPrepareCLCode(const char *code, const char *flags)
{
    gmac::enterGmac();
    Mode &mode = Mode::current();
    gmacError_t ret = mode.prepareCLCode(code, flags);
    gmac::exitGmac();

    return ret;
}

gmacError_t APICALL __oclPrepareCLBinary(const unsigned char *binary, size_t size, const char *flags)
{
    gmac::enterGmac();
    Mode &mode = Mode::current();
    gmacError_t ret = mode.prepareCLBinary(binary, size, flags);
    gmac::exitGmac();

    return ret;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
