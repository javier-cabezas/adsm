#ifndef GMAC_HAL_CPU_DEVICE_H_
#define GMAC_HAL_CPU_DEVICE_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal { namespace cpu {

class coherence_domain;

typedef hal::detail::aspace hal_aspace;
typedef hal::detail::device hal_device;
typedef hal::detail::platform hal_platform;

class platform;

class GMAC_LOCAL device :
    public hal_device,
    public util::unique<device>,
    public gmac::util::mutex<device> {

    friend class aspace;
    friend list_platform hal::get_platforms();

    typedef hal_device parent;
    typedef gmac::util::mutex<device> lock;

protected:
    GmacDeviceInfo info_;
    bool isInfoInitialized_;

public:
    device(hal_platform &plat, coherence_domain &coherenceDomain);

    hal_aspace *create_aspace(const set_siblings &siblings, gmacError_t &err);
    gmacError_t destroy_aspace(hal_aspace &as);

    hal_stream *create_stream(hal_aspace &as);
    gmacError_t destroy_stream(hal_stream &stream);

    size_t get_total_memory() const;
    size_t get_free_memory() const;

    bool has_direct_copy(const hal_device &dev) const;

    gmacError_t get_info(GmacDeviceInfo &info);
};


}}}

#include "device-impl.h"

#endif /* GMAC_HAL_CPU_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
