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
aspace::create_vaspace(detail::virt::aspace::set_processing_unit &compatibleUnits, gmacError_t &err)
{
    return virt::aspace::create<virt::aspace>(compatibleUnits, *this, err);
}

gmacError_t
aspace::destroy_vaspace(detail::virt::aspace &as)
{
    virt::aspace::destroy(as);

    return gmacSuccess;
}

}}}}


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
