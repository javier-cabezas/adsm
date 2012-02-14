#ifndef GMAC_HAL_DEVICE_H_
#define GMAC_HAL_DEVICE_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal { namespace detail {

class aspace;
class platform;
class stream;

class coherence_domain;

class GMAC_LOCAL device {
public:
    typedef std::set<device *> set_siblings;
    const static set_siblings None;

    enum type {
        DEVICE_TYPE_CPU = 0,
        DEVICE_TYPE_GPU = 1
    };
protected:
    platform &platform_;
    coherence_domain &coherenceDomain_;
    bool integrated_;
    type type_;

    device(type t, platform &platform,
                   coherence_domain &coherenceDomain);
public:
    virtual aspace *create_aspace(const set_siblings &siblings, gmacError_t &err) = 0;
    virtual gmacError_t destroy_aspace(aspace &as) = 0;

    virtual stream *create_stream(aspace &as) = 0;
    virtual gmacError_t destroy_stream(stream &stream) = 0;

    platform &get_platform();
    const platform &get_platform() const;

    coherence_domain &get_coherence_domain();
    const coherence_domain &get_coherence_domain() const;

    virtual bool has_direct_copy(const device &dev) const = 0;
    bool is_integrated() const;
    type get_type() const;

    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    virtual gmacError_t get_info(GmacDeviceInfo &info) = 0;
};

}}}

#include "device-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
