#ifndef GMAC_CORE_HPE_KERNEL_IMPL_H_
#define GMAC_CORE_HPE_KERNEL_IMPL_H_

#include "core/hpe/Process.h"

namespace __impl { namespace core { namespace hpe {

inline
Kernel::Kernel(const KernelDescriptor & k) :
    KernelDescriptor(k.getName(), k.key())
{
}

#ifdef DEBUG
inline
KernelLaunch::KernelLaunch(Mode &mode, gmac_kernel_id_t k) :
    mode_(mode), k_(k)
#else
inline
KernelLaunch::KernelLaunch(Mode &mode) :
    mode_(mode)
#endif
{ }


inline
Mode &
KernelLaunch::getMode()
{
    return mode_;
}

#ifdef DEBUG
inline
gmac_kernel_id_t
KernelLaunch::getKernelId() const
{
    return k_;
}
#endif

}}}

#endif
