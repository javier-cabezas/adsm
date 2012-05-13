#include <algorithm>

#include "util/misc.h"

#include "hal/detail/types.h"

#include "platform.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

platform::set_memory
platform::get_memories(const set_processing_unit &pUnits) const
{
    set_memory ret;

    if (pUnits.size() > 0) {
        set_aspace aspaces;
        for (processing_unit * pUnit : pUnits ) {
            aspaces.insert(&pUnit->get_paspace());
        }

        ret = get_memories(aspaces);
    }

    return ret;
}

platform::set_memory
platform::get_memories(const set_aspace &aspaces) const
{
    set_memory ret;

    if (aspaces.size() > 0) {
        const set_memory &memories = (*aspaces.begin())->get_memories();
        ret.insert(memories.begin(), memories.end());

        for (auto as : aspaces) {
            const set_memory &memories2 = as->get_memories();
            set_memory intersection;
            std::set_intersection(ret.begin(), ret.end(),
                                  memories2.begin(), memories2.end(),
                                  std::inserter(intersection, intersection.begin()));
            ret = intersection;

            if (ret.size() == 0) break;
        }
    }

    return ret;
}

platform::set_processing_unit
platform::get_processing_units(processing_unit::type type) const
{
    set_processing_unit ret;

    for (auto pUnit : pUnits_) {
        if (pUnit->get_type() == type) {
            ret.insert(pUnit);
        }
    }

    return ret;
}

platform::set_aspace
platform::get_paspaces(memory &mem) const
{
    set_aspace ret;

    for (auto as : aspaces_) {
        const aspace::set_memory &mems = as->get_memories();
        if (mems.find(&mem) != mems.end()) {
            ret.insert(as);
        }
    }

    return ret;
}

platform::set_aspace
platform::get_paspaces(processing_unit::type type) const
{
    set_aspace ret;

    for (auto pu : pUnits_) {
        if (pu->get_type() != type) {
            ret.insert(&pu->get_paspace());
        }
    }

    return ret;
}



virt::object *
platform::create_object(const memory &mem, size_t size, hal::error &err)
{
    if (memories_.find(&mem) == memories_.end()) {
        err = HAL_ERROR_INVALID_VALUE;
        return NULL;
    }

    err = HAL_SUCCESS;
    //return new object(mem, size);
    return create(mem, size);
}

hal::error
platform::destroy_object(virt::object &obj)
{
    destroy(obj);
    
    return HAL_SUCCESS;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
