#ifndef GMAC_CORE_HPE_KERNEL2_IMPL_H_
#define GMAC_CORE_HPE_KERNEL2_IMPL_H_

#include "core/hpe/process.h"

namespace __impl { namespace core { namespace hpe {

inline
const kernel::arg_list::list_object_info &
kernel::arg_list::get_objects() const
{
    return usedObjects_;
}

inline
void
kernel::arg_list::add_object(hostptr_t ptr, /* unsigned index, */GmacProtection prot)
{
    TRACE(LOCAL, "Adding object to argument list");
    // NOTE:
    // Path used by OpenCL, since KernelLaunch objects can be reused
    usedObjects_.push_back(memory::ObjectInfo(ptr, prot));
}

inline
kernel::kernel(const hal::kernel_t &parent) :
    hal::kernel_t(parent)
{
}

inline
kernel::launch::launch(kernel &parent, vdevice &dev, hal::kernel_t::config &conf,
                                                     hal::kernel_t::arg_list &args, 
                                                     hal::stream_t &stream, gmacError_t &ret) :
    hal::kernel_t::launch::launch(parent, conf, args, stream),
    dev_(dev)
{
    ret = gmacSuccess;
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
hal::event_t
kernel::launch::get_event()
{
    return event_;
}

inline
const kernel::arg_list &
kernel::launch::get_arg_list()
{
    return reinterpret_cast<const kernel::arg_list &>(Parent::get_arg_list());
}

inline
kernel::launch *
kernel::launch_config(vdevice &dev, hal::kernel_t::config &conf, hal::kernel_t::arg_list &args, hal::stream_t &stream, gmacError_t &err)
{
    launch *ret = NULL;

    ret = new launch(*this, dev, conf, args, stream, err);

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
const std::list<memory::ObjectInfo> &
KernelLaunch::getObjects() const
{
    return usedObjects_;
}

#endif

}}}

#endif
