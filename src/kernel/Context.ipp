#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

namespace gmac {

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
