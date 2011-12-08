#ifndef GMAC_HAL_COHERENCE_DOMAIN_H_
#define GMAC_HAL_COHERENCE_DOMAIN_H_

#include <set>

#include "config/common.h"

namespace __impl { namespace hal {

namespace detail {

template <typename D> 
class GMAC_LOCAL coherence_domain :
    D::SetSiblings {
    typedef typename D::SetSiblings Parent;
protected:
    coherence_domain();

public:
    gmacError_t add_device(D &device);
    gmacError_t create_stream(D &device);

    size_t size();

    typename D::SetSiblings &get_devices();
    const typename D::SetSiblings &get_devices() const;
};

}

}}

#include "coherence_domain-impl.h"

#endif /* GMAC_HAL_DOMAIN_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
