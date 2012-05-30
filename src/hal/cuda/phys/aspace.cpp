#include "hal/cuda/types.h"

#include "aspace.h"
#include "processing_unit.h"

namespace __impl { namespace hal { namespace cuda { namespace phys {

virt::hal_aspace *
aspace::create_vaspace(aspace::set_processing_unit &compatibleUnits, hal::error &err)
{
    //ASSERTION(get_processing_units().find(&pu) != get_processing_units().end());

    return virt::aspace::observe_construct::create<virt::aspace>(compatibleUnits, *this, err);
}

hal::error
aspace::destroy_vaspace(virt::hal_aspace &_as)
{
    virt::aspace &as = reinterpret_cast<virt::aspace &>(_as);
    virt::aspace::observe_destruct::destroy(as);

    return hal::error::HAL_SUCCESS;
}

}}}}
