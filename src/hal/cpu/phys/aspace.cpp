#include "hal/types.h"

namespace __impl { namespace hal { namespace cpu { namespace phys {

aspace::aspace(detail::phys::platform &plat, const set_memory &memories) :
    parent(plat, memories)
{
}

aspace::~aspace()
{
}

detail::virt::aspace *
aspace::create_vaspace(detail::virt::aspace::set_processing_unit &compatibleUnits, hal::error &err)
{
    return virt::aspace::create<virt::aspace>(compatibleUnits, *this, err);
}

hal::error
aspace::destroy_vaspace(detail::virt::aspace &as)
{
    virt::aspace::destroy(as);

    return hal::error::HAL_SUCCESS;
}

}}}}


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
