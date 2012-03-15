#ifndef GMAC_HAL_PLATFORM_H_
#define GMAC_HAL_PLATFORM_H_

#include <set>

#include "config/common.h"

#include "util/unique.h"

#include "processing_unit.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

class coherence_domain;

class GMAC_LOCAL platform :
    public util::unique<platform>
{
public:
    typedef std::set<processing_unit *> set_processing_unit;
    typedef std::set<aspace_ptr>        set_aspace;
    typedef std::set<memory_ptr>        set_memory;

private:
    set_processing_unit pUnits_;
    set_memory memories_;
    set_aspace aspaces_;
    
public:
    virtual ~platform();
    void add_processing_unit(processing_unit &d);
    void add_paspace(aspace_ptr as);

    const set_processing_unit &get_processing_units() const;
    set_processing_unit get_processing_units(processing_unit::type type);

    const set_memory &get_memories() const;
    const set_aspace &get_paspaces() const;
};

}

}}}

#include "platform-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
