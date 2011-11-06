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
    typedef std::set<Current *> SetSiblings;
    const static SetSiblings None;
protected:
    typename I::coherence_domain &coherenceDomain_;

    device(typename I::coherence_domain &coherenceDomain);

    bool integrated_;

public:
    virtual typename I::context *create_context(const SetSiblings &siblings = None) = 0;
    virtual gmacError_t destroy_context(typename I::context &context) = 0;

    virtual typename I::stream *create_stream(typename I::context &context) = 0;
    virtual gmacError_t destroy_stream(typename I::stream &stream) = 0;

    typename I::coherence_domain &get_coherence_domain();
    const typename I::coherence_domain &get_coherence_domain() const;

    virtual bool has_direct_copy(const device &dev) const = 0;
    bool is_integrated() const;
};

template <typename I>
const typename device<I>::SetSiblings device<I>::None;

}

}}

#include "device-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
