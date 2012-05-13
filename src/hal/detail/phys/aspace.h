#ifndef GMAC_HAL_DETAIL_PHYS_ASPACE_H_
#define GMAC_HAL_DETAIL_PHYS_ASPACE_H_

#include <set>

#include "hal/error.h"

#include "util/unique.h"

namespace __impl { namespace hal { namespace detail {

namespace virt {
class aspace;
}

namespace phys {

class platform;
class processing_unit;

class GMAC_LOCAL aspace :
    public util::unique<aspace> {
public:
    typedef std::set<const memory *> set_memory;
    typedef std::set<processing_unit *> set_processing_unit;

private:
    platform &platform_;
    set_memory memories_;
    set_processing_unit pUnits_;

public:
    aspace(platform &plat, const set_memory &memories) :
        platform_(plat),
        memories_(memories)
    {
        TRACE(LOCAL, FMT_ID2" Create", get_print_id2());
    }

    virtual ~aspace()
    {
        TRACE(LOCAL, FMT_ID2" Destroy", get_print_id2());
    }
    
    virtual virt::aspace *create_vaspace(aspace::set_processing_unit &compatibleUnits, hal::error &err) = 0;
    virtual hal::error destroy_vaspace(virt::aspace &as) = 0;

    void add_processing_unit(processing_unit &pUnit)
    {
        pUnits_.insert(&pUnit);
    }

    const set_memory &get_memories() const
    {
        return memories_;
    }

    const set_processing_unit &get_processing_units() const
    {
        return pUnits_;
    }

    const platform &get_platform() const;
};

}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
