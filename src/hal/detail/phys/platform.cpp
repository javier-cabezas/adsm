#include "platform.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

platform::set_processing_unit
platform::get_processing_units(processing_unit::type type)
{
    set_processing_unit ret;

    for (auto pUnit : pUnits_) {
        if (pUnit->get_type() == type) {
            ret.insert(pUnit);
        }
    }

    return ret;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
