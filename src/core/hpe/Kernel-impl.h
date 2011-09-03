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


inline
void
KernelLaunch::addObject(hostptr_t ptr, unsigned index)
{
    // NOTE:
    // Path used by OpenCL, since KernelLaunch objects can be reused
    std::map<unsigned, std::list<hostptr_t>::iterator>::iterator itMap = paramToParamPtr_.find(index);
    if (itMap == paramToParamPtr_.end()) {
        usedObjects_.push_back(ptr);
        std::list<hostptr_t>::iterator iter = --(usedObjects_.end());
        paramToParamPtr_.insert(std::map<unsigned, std::list<hostptr_t>::iterator>::value_type(index, iter));
    } else {
        std::list<hostptr_t>::iterator iter = itMap->second;
        (*iter) = ptr;
    }
}

inline
const std::list<hostptr_t> &
KernelLaunch::getObjects() const
{
    return usedObjects_;
}

}}}

#endif
