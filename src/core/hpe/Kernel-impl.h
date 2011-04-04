#ifndef GMAC_CORE_HPE_KERNEL_IMPL_H_
#define GMAC_CORE_HPE_KERNEL_IMPL_H_

namespace __impl { namespace core { namespace hpe {

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

}}}

#endif
