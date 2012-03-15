#include "processing_unit.h"
#include "hal/detail/types.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

processing_unit::processing_unit(type t, platform &platform,
                                 set_memory_connection &memories,
                                 set_aspace &aspaces) :
    platform_(platform),
    type_(t),
    memories_(memories),
    aspaces_(aspaces)
{
    for (auto a : aspaces_) {
        a->add_processing_unit(*this);
    }

    for (auto m : memories_) {
        m.mem->add_attached_unit(*this);
    }
}

}}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
