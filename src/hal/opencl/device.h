#ifndef GMAC_HAL_OPENCL_DEVICE_H_
#define GMAC_HAL_OPENCL_DEVICE_H_

#include "hal/device.h"
#include "util/unique.h"

#include "helper/opencl_helper.h"

#include "types.h"

namespace __impl { namespace hal { namespace opencl {

class coherence_domain;

typedef hal::detail::device<implementation_traits> hal_device;

class device;

class GMAC_LOCAL platform {
    cl_platform_id openclPlatformId_;
    std::list<device *> devices_;

    cl_context ctx_;

public:
    platform(cl_platform_id id, cl_context ctx) :
        openclPlatformId_(id),
        ctx_(ctx)
    {
    }

    cl_platform_id
    get_cl_platform_id() const
    {
        return openclPlatformId_;
    }

    void
    add_device(device &d)
    {
        devices_.push_back(&d);
    }

    inline unsigned
    get_ndevices()
    {
        return devices_.size();
    }

    cl_device_id *get_cl_device_array();

    inline cl_context
    get_context()
    {
        return ctx_;
    }
};

class GMAC_LOCAL device :
    public hal_device,
    public util::unique<device> {
    typedef hal_device Parent;
    friend class platform;
protected:
    platform &platform_;

    cl_device_id openclDeviceId_;
    cl_context context_;

    helper::opencl_version openclVersion_;

    size_t memorySize_;

    bool integrated_;

public:
    device(platform &p,
           cl_device_id openclDeviceId,
           coherence_domain &coherenceDomain);

    context_t *create_context(const SetSiblings &siblings, gmacError_t &err);
    gmacError_t destroy_context(context_t &context);

    stream_t *create_stream(context_t &context);
    gmacError_t destroy_stream(stream_t &stream);

    size_t get_total_memory() const;
    size_t get_free_memory() const;

    bool has_direct_copy(const Parent &dev) const;

    platform &get_platform()
    {
        return platform_;
    }

    const platform &get_platform() const
    {
        return platform_;
    }

#if 0
    static void set_devices(std::list<opencl::device *> devices)
    {
        device::Devices_ = devices;
    }

    static std::list<opencl::device *> Devices_;
#endif
};

}}}

#endif /* GMAC_HAL_OPENCL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
