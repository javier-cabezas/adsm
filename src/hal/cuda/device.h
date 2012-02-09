#ifndef GMAC_HAL_CUDA_DEVICE_H_
#define GMAC_HAL_CUDA_DEVICE_H_

#include "hal/device.h"
#include "util/unique.h"

#include "types.h"

namespace __impl { namespace hal { namespace cuda {

class coherence_domain;

typedef hal::detail::device<implementation_traits> hal_device;

class device;

class platform;


class GMAC_LOCAL device :
    public hal_device,
    public util::unique<device>,
    public gmac::util::mutex<device> {
    friend class aspace;

    friend list_platform hal::get_platforms();

    typedef hal_device parent;

protected:
    device(parent::type t, platform &plat, coherence_domain &coherenceDomain);

#if 0
    aspace *create_context(const set_siblings &siblings, gmacError_t &err);
    gmacError_t destroy_context(aspace &context);

    stream_t *create_stream(aspace &context);
    gmacError_t destroy_stream(stream_t &stream);

    bool has_direct_copy(const device &dev) const;

    gmacError_t get_info(GmacDeviceInfo &info);
#endif
};


class GMAC_LOCAL gpu :
    public device {
    friend class aspace;

    typedef device parent;
protected:
    CUdevice cudaDevice_;

    int major_;
    int minor_;

    GmacDeviceInfo info_;
    bool isInfoInitialized_;

public:
    gpu(CUdevice cudaDevice, platform &plat, coherence_domain &coherenceDomain);

    aspace *create_context(const set_siblings &siblings, gmacError_t &err);
    gmacError_t destroy_context(aspace &context);

    stream_t *create_stream(aspace &context);
    gmacError_t destroy_stream(stream_t &stream);

    int get_major() const;
    int get_minor() const;

    size_t get_total_memory() const;
    size_t get_free_memory() const;

    bool has_direct_copy(const device &dev) const;

    gmacError_t get_info(GmacDeviceInfo &info);
};

class GMAC_LOCAL cpu :
    public device {
    typedef device parent;
public:
    cpu(platform &plat, coherence_domain &coherenceDomain);

    aspace *create_context(const set_siblings &siblings, gmacError_t &err);

    gmacError_t destroy_context(aspace &context);

    stream_t *create_stream(aspace &context);

    gmacError_t destroy_stream(stream_t &stream);

    size_t get_total_memory() const;

    size_t get_free_memory() const;

    bool has_direct_copy(const device &dev) const;

    gmacError_t get_info(GmacDeviceInfo &info);
};

}}}

#endif /* GMAC_HAL_CUDA_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
