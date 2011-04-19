#ifndef GMAC_CORE_KERNEL_IMPL_H_
#define GMAC_CORE_KERNEL_IMPL_H_

#include "Mode.h"

namespace __impl { namespace core {

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.getName(), k.key())
{
}

inline
KernelLaunch::KernelLaunch(__impl::core::Mode &mode) :
    mode_(mode)
{
}

inline
Mode &
KernelLaunch::getMode()
{
    return mode_;
}

}}

#endif
