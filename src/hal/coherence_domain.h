#ifndef GMAC_HAL_COHERENCE_DOMAIN_H_
#define GMAC_HAL_COHERENCE_DOMAIN_H_

#include <set>

#include "config/common.h"

#include "device.h"

namespace __impl { namespace hal {

namespace detail {

class GMAC_LOCAL coherence_domain :
    device::set_siblings {
    typedef device::set_siblings Parent;
protected:
    coherence_domain();

public:
    gmacError_t add_device(device &device);
    gmacError_t create_stream(device &device);

    size_t size();

    device::set_siblings &get_devices();
    const device::set_siblings &get_devices() const;
};

}

}}

#include "coherence_domain-impl.h"

#endif /* GMAC_HAL_DOMAIN_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
