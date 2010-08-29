#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

namespace gmac {

inline
void Context::kernel(gmacKernel_t k, Kernel * kernel)
{
    assertion(kernel != NULL);
    trace("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = kernels.find(k);
    assertion(i == kernels.end());
    kernels[k] = kernel;
}

inline
Kernel * Context::kernel(gmacKernel_t k)
{
    KernelMap::iterator i;
    i = kernels.find(k);
    if (i != kernels.end()) {
        return i->second;
    }
    return NULL;
}

inline
gmacError_t Context::copyToDevice(void *dev, const void *host, size_t size)
{
    return acc->copyToDevice(dev, host, size);
}

inline
gmacError_t Context::copyToHost(void *host, const void *dev, size_t size)
{
    return acc->copyToHost(host, dev, size);
}

inline
gmacError_t Context::copyDevice(void *dst, const void *src, size_t size)
{
    return acc->copyDevice(dst, src, size);
}


}

#endif
