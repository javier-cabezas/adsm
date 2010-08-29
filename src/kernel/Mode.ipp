#ifndef __KERNEL_MODE_IPP_
#define __KERNEL_MODE_IPP_

#include "Context.h"

namespace gmac {

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
    gmacError_t ret = acc->malloc(addr, size, align);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::free(void *addr)
{
    switchIn();
    gmacError_t ret = acc->free(addr);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyToDevice(void *dev, const void *host, size_t size)
{
    switchIn();
    gmacError_t ret = context->copyToDevice(dev, host, size);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyToHost(void *host, const void *dev, size_t size)
{
    switchIn();
    gmacError_t ret = context->copyToHost(host, dev, size);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::copyDevice(void *dst, const void *src, size_t size)
{
    switchIn();
    gmacError_t ret = context->copyDevice(dst, src, size);
    switchOut();
    return ret;
}

inline
gmacError_t Mode::sync()
{
    switchIn();
    gmacError_t ret = context->sync();
    switchOut();
    return ret;
}


inline
gmac::KernelLaunch *Mode::launch(gmacKernel_t kernel)
{
    switchIn();
    gmac::KernelLaunch *ret = context->launch(kernel);
    switchOut();
    return ret;
}


}

#endif
