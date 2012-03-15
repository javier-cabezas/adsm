#ifndef GMAC_HAL_CUDA_PLATFORM_H_
#define GMAC_HAL_CUDA_PLATFORM_H_

#include "hal/detail/phys/platform.h"

namespace __impl { namespace hal { namespace cuda { namespace phys {

typedef hal::detail::phys::platform hal_platform;

class GMAC_LOCAL platform :
    public hal_platform
{
    typedef hal_platform parent;
};

}}}}

#endif /* PLATFORM_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
