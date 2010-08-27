#ifndef __KERNEL_CONTEXT_IPP_
#define __KERNEL_CONTEXT_IPP_

namespace gmac {

inline
void Context::kernel(gmacKernel_t k, Kernel * kernel)
{
    assertion(kernel != NULL);
    trace("CTX: %p Registering kernel %s: %p", this, kernel->name(), k);
    KernelMap::iterator i;
    i = _kernels.find(k);
    assertion(i == _kernels.end());
    _kernels[k] = kernel;
}

inline
Kernel * Context::kernel(gmacKernel_t k)
{
    KernelMap::iterator i;
    i = _kernels.find(k);
    if (i != _kernels.end()) {
        return i->second;
    }
    return NULL;
}


}

#endif
