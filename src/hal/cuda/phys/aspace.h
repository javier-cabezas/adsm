#ifndef GMAC_HAL_CUDA_PHYS_ASPACE_H_
#define GMAC_HAL_CUDA_PHYS_ASPACE_H_

#include "hal/cuda/types.h"

namespace __impl { namespace hal { namespace cuda {

namespace virt {
typedef hal::detail::virt::aspace hal_aspace;
}

namespace phys {

typedef hal::detail::phys::aspace hal_aspace;
typedef hal::detail::phys::platform hal_platform;
typedef hal::detail::phys::processing_unit hal_processing_unit;

class GMAC_LOCAL aspace :
    public hal_aspace {
public:
    aspace(hal_platform &plat, const hal_aspace::set_memory &memories) :
        hal_aspace(plat, memories)
    {
    }

    virt::hal_aspace *create_vaspace(aspace::set_processing_unit &compatibleUnits, hal::error &err);
    hal::error destroy_vaspace(virt::hal_aspace &as);
};

typedef util::shared_ptr<aspace>       aspace_ptr;
typedef util::shared_ptr<const aspace> aspace_const_ptr;

}

}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
