#include "hal/cpu/types.h"

namespace __impl { namespace hal { namespace cpu { namespace phys {

processing_unit::processing_unit(detail::phys::platform &platform, detail::phys::aspace &as) :
    parent(platform, parent::PUNIT_TYPE_CPU, as)
{
}

processing_unit::~processing_unit()
{}

hal::error
processing_unit::get_info(GmacDeviceInfo &info)
{
    return HAL_SUCCESS;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
