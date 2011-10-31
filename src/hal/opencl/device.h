#ifndef GMAC_HAL_OPENCL_DEVICE_H_
#define GMAC_HAL_OPENCL_DEVICE_H_

#include "hal/device.h"
#include "hal/opencl/util/opencl_util.h"
#include "util/Unique.h"

#include "types.h"

namespace __impl { namespace hal { namespace opencl {

class coherence_domain;

typedef hal::detail::device<coherence_domain, aspace_t, stream_t, event_t, async_event_t> hal_device;

class device :
    public hal_device,
    public __impl::util::Unique<device> {
    typedef hal_device Parent;
protected:
    cl_device_id openclDeviceId_;
    cl_platform_id openclPlatformId_;

    util::opencl_version openclVersion_;

    size_t memorySize_;

    bool integrated_;

public:
    device(cl_device_id openclDeviceId,
           cl_platform_id openclPlatformId,
           coherence_domain &coherenceDomain);

    aspace_t create_address_space(const Parent::SetSiblings &siblings = Parent::None);
    stream_t create_stream(aspace_t &aspace);

    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream);

    async_event_t copy_async(accptr_t dst, hostptr_t src, size_t count, stream_t &stream);
    async_event_t copy_async(hostptr_t dst, accptr_t src, size_t count, stream_t &stream);
    async_event_t copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream);

    gmacError_t sync(async_event_t &event);
    gmacError_t sync(stream_t &stream);
};

}}}

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
