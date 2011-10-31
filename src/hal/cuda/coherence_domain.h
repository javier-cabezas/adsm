#ifndef GMAC_HAL_CUDA_COHERENCE_DOMAIN_H_
#define GMAC_HAL_CUDA_COHERENCE_DOMAIN_H_

#include "hal/coherence_domain.h"
#include "util/Unique.h"

#include "types.h"

#include "device.h"

namespace __impl { namespace hal { namespace cuda {

class coherence_domain :
    public hal::detail::coherence_domain<device>,
    public util::unique<coherence_domain> {
public:
    coherence_domain();
};

}}}

#endif /* GMAC_HAL_COHERENCE_DOMAIN_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
