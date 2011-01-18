#ifdef USE_DBC

#include "core/Context.h"

namespace __dbc { namespace core {

Context::Context(__impl::core::Accelerator &acc, unsigned id) :
    __impl::core::Context(acc, id)
{
}

Context::~Context()
{
}

gmacError_t
Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    REQUIRES(acc != NULL);
    REQUIRES(host != NULL);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::Context::copyToAccelerator(acc, host, size);
    return ret;
}

gmacError_t
Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    REQUIRES(host != NULL);
    REQUIRES(acc != NULL);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::Context::copyToHost(host, acc, size);
    return ret;
}

gmacError_t
Context::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    REQUIRES(src != NULL);
    REQUIRES(dst != NULL);
    REQUIRES(size > 0);
    gmacError_t ret;
    ret = __impl::core::Context::copyAccelerator(dst, src, size);
    return ret;
}

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
