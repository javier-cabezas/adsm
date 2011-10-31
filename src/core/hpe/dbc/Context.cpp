#ifdef USE_DBC

#include "core/hpe/Context.h"

namespace __dbc { namespace core { namespace hpe {

Context::Context(__impl::core::hpe::Mode &mode, __impl::hal::stream_t &streamLaunch,
                                                __impl::hal::stream_t &streamToAccelerator,
                                                __impl::hal::stream_t &streamToHost,
                                                __impl::hal::stream_t &streamAccelerator) :
    __impl::core::hpe::Context(mode, streamLaunch, streamToAccelerator, streamToHost, streamAccelerator)
{
}

Context::~Context()
{
}

#if 0
gmacError_t
Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    REQUIRES(acc != accptr_t(0));
    REQUIRES(host != NULL);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::hpe::Context::copyToAccelerator(acc, host, size);
    return ret;
}

gmacError_t
Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    REQUIRES(host != NULL);
    REQUIRES(acc != accptr_t(0));
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::hpe::Context::copyToHost(host, acc, size);
    return ret;
}
#endif

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
