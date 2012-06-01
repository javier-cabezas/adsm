#ifndef GMAC_HAL_CPU_PHYS_PROCESSING_UNIT_H_
#define GMAC_HAL_CPU_PHYS_PROCESSING_UNIT_H_

#include <set>

#include "config/common.h"
#include "util/unique.h"

#include "memory.h"

namespace __impl { namespace hal {

namespace cpu {
    
namespace virt {
    class aspace;
    class scheduler;
}

class stream;

namespace phys {

typedef detail::phys::processing_unit hal_processing_unit;

class aspace;
class coherence_domain;
class platform;

class GMAC_LOCAL processing_unit :
    public hal_processing_unit {

    typedef hal_processing_unit parent;

public:
    processing_unit(detail::phys::platform &platform, detail::phys::aspace &as, virt::scheduler &sched);
    virtual ~processing_unit();

    hal::error get_info(GmacDeviceInfo &info);
};

}}}}

#endif /* GMAC_HAL_PHYS_PROCESSING_UNIT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
