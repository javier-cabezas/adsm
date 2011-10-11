#ifndef GMAC_HAL_DEVICE_H_
#define GMAC_HAL_DEVICE_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

template <typename D> 
class coherence_domain;

template <typename CD, typename A, typename S, typename E, typename AE> 
class device {
public:
    typedef device<CD, A, S, E, AE> Current;
    typedef std::set<Current *> SetSiblings;
    const static SetSiblings None;
protected:
    CD &coherenceDomain_;

    device(CD &coherenceDomain);

public:
    virtual A create_address_space(const SetSiblings &siblings = None) = 0;
    virtual S create_stream(S &aspace) = 0;

    virtual E copy(accptr_t dst, hostptr_t src, size_t count, S &stream) = 0;
    virtual E copy(hostptr_t dst, accptr_t src, size_t count, S &stream) = 0;
    virtual E copy(accptr_t dst, accptr_t src, size_t count, S &stream) = 0;

    virtual AE copy_async(accptr_t dst, hostptr_t src, size_t count, S &stream) = 0;
    virtual AE copy_async(hostptr_t dst, accptr_t src, size_t count, S &stream) = 0;
    virtual AE copy_async(accptr_t dst, accptr_t src, size_t count, S &stream) = 0;

    virtual gmacError_t sync(AE &event) = 0;
    virtual gmacError_t sync(S &stream) = 0;

    CD &get_coherence_domain();
};
}

}}

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
