#ifndef GMAC_HAL_CUDA_COHERENCE_DOMAIN_H_
#define GMAC_HAL_CUDA_COHERENCE_DOMAIN_H_

#include "hal/CoherenceDomain.h"
#include "util/Unique.h"

#include "types.h"

namespace __impl { namespace hal { namespace cuda {

class CoherenceDomain :
    public hal::detail::CoherenceDomain<Device>,
    public util::Unique<CoherenceDomain> {
public:
    CoherenceDomain();
};

}}}

#endif /* GMAC_HAL_COHERENCE_DOMAIN_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
