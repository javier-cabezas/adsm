#ifdef USE_DBC

#include "core/hpe/Context.h"

namespace __dbc { namespace core { namespace hpe {

Context::Context(__impl::core::hpe::Accelerator &acc, __impl::core::hpe::Mode &mode, unsigned id) :
    __impl::core::hpe::Context(acc, mode, id)
{
}

Context::~Context()
{
}

gmacError_t
Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    REQUIRES(acc != 0);
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
    REQUIRES(acc != 0);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::hpe::Context::copyToHost(host, acc, size);
    return ret;
}

gmacError_t
Context::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    REQUIRES(src != 0);
    REQUIRES(dst != 0);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::hpe::Context::copyAccelerator(dst, src, size);
    return ret;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
