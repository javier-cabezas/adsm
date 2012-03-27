#ifndef GMAC_HAL_PLATFORM_H_
#define GMAC_HAL_PLATFORM_H_

#include <set>

#include "config/common.h"

#include "util/factory.h"
#include "util/unique.h"

#include "processing_unit.h"

namespace __impl { namespace hal { namespace detail {

namespace virt {
    class object;
}

namespace phys {

class coherence_domain;

class GMAC_LOCAL platform :
    public util::unique<platform>,
    protected util::factory<virt::object>
{
public:
    typedef std::set<const memory *>    set_memory;
    typedef std::set<processing_unit *> set_processing_unit;
    typedef std::set<aspace *>          set_aspace;

private:
    set_memory memories_;
    set_processing_unit pUnits_;
    set_aspace aspaces_;
    
public:
    virtual ~platform();

    void add_processing_unit(processing_unit &d);
    void add_paspace(aspace &as);

    const set_memory &get_memories() const;
    set_memory get_memories(const set_processing_unit &pUnits) const;
    set_memory get_memories(const set_aspace &aspaces) const;

    const set_processing_unit &get_processing_units() const;
    set_processing_unit get_processing_units(processing_unit::type type) const;

    const set_aspace &get_paspaces() const;
    set_aspace get_paspaces(memory &mem) const;
    set_aspace get_paspaces(processing_unit::type type) const;

    virt::object *create_object(const memory &mem, size_t size, gmacError_t &err);
    gmacError_t destroy_object(virt::object &obj);
};

}

}}}

#include "platform-impl.h"

#endif /* GMAC_HAL_DEVICE_H_ */


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
