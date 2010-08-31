#ifndef __KERNEL_MODE_IPP_
#define __KERNEL_MODE_IPP_

#include "Context.h"

namespace gmac {

inline
void Mode::kernel(gmacKernel_t k, Kernel * kernel)
{
    assertion(kernel != NULL);
    trace("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = kernels.find(k);
    assertion(i == kernels.end());
    kernels[k] = kernel;
}


inline 
Mode *Mode::current()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == NULL) mode = proc->create();
    gmac::util::Logger::ASSERTION(mode != NULL);
    return mode;
}


inline
void Mode::attach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode == this) return;
    if(mode != NULL) mode->destroy();
    key.set(this);
    count++;
}

inline
void Mode::detach()
{
    Mode *mode = static_cast<Mode *>(Mode::key.get());
    if(mode != NULL) mode->destroy();
    key.set(NULL);
}

inline
gmacError_t Mode::malloc(void **addr, size_t size, unsigned align)
{
    switchIn();
    __error = acc->malloc(addr, size, align);
    switchOut();
    return __error;
}

inline
gmacError_t Mode::free(void *addr)
{
    switchIn();
    __error = acc->free(addr);
    switchOut();
    return __error;
}

inline
gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    switchIn();
    __error = context->copyToDevice(dev, host, size);
    switchOut();
    return __error;
}

inline
gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    switchIn();
    __error = context->copyToHost(host, dev, size);
    switchOut();
    return __error;
}

inline
gmacError_t Mode::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    __error = context->copyDevice(dst, src, size);
    switchOut();
    return __error;
}

inline
gmacError_t Mode::memset(void *addr, int c, size_t size)
{
    switchIn();
    __error = context->memset(addr, c, size);
    switchOut();
    return __error;
}

inline
gmac::KernelLaunch *Mode::launch(const char *kernel)
{
    KernelMap::iterator i = kernels.find(kernel);
    assert(i != kernels.end());
    gmac::Kernel * k = i->second;
    assertion(k != NULL);
    switchIn();
    gmac::KernelLaunch *l  = context->launch(k);
    switchOut();

    return l;
}

inline
gmacError_t Mode::sync()
{
    switchIn();
    __error = context->sync();
    switchOut();
    return __error;
}

}


#endif
