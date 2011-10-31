#ifndef GMAC_CORE_HPE_KERNEL2_IMPL_H_
#define GMAC_CORE_HPE_KERNEL2_IMPL_H_

#include "core/hpe/process.h"

namespace __impl { namespace core { namespace hpe {

inline
kernel::kernel(const hal::kernel_t &parent) :
    hal::kernel_t(parent)
{
}

inline
kernel::launch::launch(kernel &parent, hal::kernel_t::config &config, vdevice &dev, hal::stream_t &stream) :
    hal::kernel_t::launch::launch(parent, config, stream),
    dev_(dev)
{
}

inline
vdevice &
kernel::launch::get_virtual_device()
{
    return dev_;
}

inline
const vdevice &
kernel::launch::get_virtual_device() const
{
    return dev_;
}

inline
const kernel::launch::list_object_info &
kernel::launch::get_objects() const
{
    return usedObjects_;
}

inline
hal::async_event_t *
kernel::launch::get_event()
{
    return event_;
}

inline
kernel::launch *
kernel::launch_config(hal::kernel_t::config &config, vdevice &dev, hal::stream_t &stream, gmacError_t &err)
{
    launch *ret = NULL;

    ret = new launch(*this, config, dev, stream);

    return ret;
}

#if 0
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
KernelLaunch::addObject(hostptr_t ptr, unsigned index, GmacProtection prot)
{
    // NOTE:
    // Path used by OpenCL, since KernelLaunch objects can be reused
    std::map<unsigned, std::list<memory::ObjectInfo>::iterator>::iterator itMap = paramToParamPtr_.find(index);
    if (itMap == paramToParamPtr_.end()) {
        usedObjects_.push_back(memory::ObjectInfo(ptr, prot));
        std::list<memory::ObjectInfo>::iterator iter = --(usedObjects_.end());
        paramToParamPtr_.insert(std::map<unsigned, std::list<memory::ObjectInfo>::iterator>::value_type(index, iter));
    } else {
        std::list<memory::ObjectInfo>::iterator iter = itMap->second;
        (*iter).first = ptr;
        (*iter).second = prot;
    }
}

inline
const std::list<memory::ObjectInfo> &
KernelLaunch::getObjects() const
{
    return usedObjects_;
}

#endif

}}}

#endif
