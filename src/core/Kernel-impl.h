#ifndef GMAC_CORE_KERNEL_IMPL_H_
#define GMAC_CORE_KERNEL_IMPL_H_

namespace __impl { namespace core {

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.name(), k.key())
{
}

inline
Kernel::~Kernel()
{
}


inline
KernelLaunch::~KernelLaunch()
{
}

}}

#endif
