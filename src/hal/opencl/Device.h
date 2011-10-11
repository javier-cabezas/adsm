#ifndef GMAC_HAL_OPENCL_DEVICE_H_
#define GMAC_HAL_OPENCL_DEVICE_H_

#include "hal/Device.h"
#include "util/Unique.h"

#include "types.h"
#include "CoherenceDomain.h"

namespace __impl { namespace hal { namespace opencl {

class Device :
    public hal::detail::Device<CoherenceDomain, aspace_t, stream_t, event_t, async_event_t>,
    public util::Unique<Device> {
    typedef hal::detail::Device<CoherenceDomain, aspace_t, stream_t, event_t, async_event_t> Parent;
protected:
    cl_device_id cudaDevice_;

    int major_;
    int minor_;

    size_t memorySize_;

    bool integrated_;

public:
    Device(cl_device_id cudaDeviceId, CoherenceDomain &coherenceDomain);

    aspace_t createPAddressSpace();
    stream_t createStream(aspace_t &aspace);

    event_t copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream);
    event_t copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream);
    event_t copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream);

    async_event_t copyAsync(accptr_t dst, hostptr_t src, size_t count, stream_t &stream);
    async_event_t copyAsync(hostptr_t dst, accptr_t src, size_t count, stream_t &stream);
    async_event_t copyAsync(accptr_t dst, accptr_t src, size_t count, stream_t &stream);

    gmacError_t sync(async_event_t &event);
    gmacError_t sync(stream_t &stream);
};

}}}

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
