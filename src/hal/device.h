#ifndef GMAC_HAL_DEVICE_H_
#define GMAC_HAL_DEVICE_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

template <typename D> 
class coherence_domain;

template <typename I> 
class GMAC_LOCAL device {
public:
    typedef device<I> Current;
    typedef std::set<Current *> set_siblings;
    const static set_siblings None;

    enum type {
        DEVICE_TYPE_CPU = 0,
        DEVICE_TYPE_GPU = 0
    };
protected:
    typename I::platform &platform_;
    typename I::coherence_domain &coherenceDomain_;
    bool integrated_;
    type type_;

    device(type t, typename I::platform &platform,
                   typename I::coherence_domain &coherenceDomain);
public:
    virtual typename I::context *create_context(const set_siblings &siblings, gmacError_t &err) = 0;
    virtual gmacError_t destroy_context(typename I::context &context) = 0;

    virtual typename I::stream *create_stream(typename I::context &context) = 0;
    virtual gmacError_t destroy_stream(typename I::stream &stream) = 0;

    typename I::platform &get_platform();
    const typename I::platform &get_platform() const;

    typename I::coherence_domain &get_coherence_domain();
    const typename I::coherence_domain &get_coherence_domain() const;

    virtual bool has_direct_copy(const typename I::device &dev) const = 0;
    bool is_integrated() const;
    type get_type() const;

    virtual size_t get_total_memory() const = 0;
    virtual size_t get_free_memory() const = 0;

    virtual gmacError_t get_info(GmacDeviceInfo &info) = 0;
};

template <typename I>
const typename device<I>::set_siblings device<I>::None;

}

}}

#include "device-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
