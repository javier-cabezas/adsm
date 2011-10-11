#ifndef GMAC_HAL_DEVICE_H_
#define GMAC_HAL_DEVICE_H_

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

template <typename C, typename A, typename S, typename E, typename AE> 
class Device {
protected:
    C &coherenceDomain_;

    Device(C &coherenceDomain);

public:
    virtual A createPAddressSpace() = 0;
    virtual S createStream(S &aspace) = 0;

    virtual E copy(accptr_t dst, hostptr_t src, size_t count, S &stream) = 0;
    virtual E copy(hostptr_t dst, accptr_t src, size_t count, S &stream) = 0;
    virtual E copy(accptr_t dst, accptr_t src, size_t count, S &stream) = 0;

    virtual AE copyAsync(accptr_t dst, hostptr_t src, size_t count, S &stream) = 0;
    virtual AE copyAsync(hostptr_t dst, accptr_t src, size_t count, S &stream) = 0;
    virtual AE copyAsync(accptr_t dst, accptr_t src, size_t count, S &stream) = 0;

    virtual gmacError_t sync(AE &event) = 0;
    virtual gmacError_t sync(S &stream) = 0;

    C &getCoherenceDomain();
};
}

}}

#endif /* GMAC_HAL_DEVICE_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
