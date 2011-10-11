#ifndef GMAC_HAL_COHERENCE_DOMAIN_H_
#define GMAC_HAL_COHERENCE_DOMAIN_H_

#include <vector>

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

template <typename D> 
class CoherenceDomain :
    std::vector<D *> {
    typedef std::vector<D *> Parent;
protected:
    CoherenceDomain();

public:
    gmacError_t addDevice(D &device);
    gmacError_t createStream(D &device);

    size_t size();
};
}

}}

#endif /* GMAC_HAL_DOMAIN_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
