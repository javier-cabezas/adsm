#ifndef GMAC_HAL_DETAIL_VIRT_ASPACE_IMPL_H_
#define GMAC_HAL_DETAIL_VIRT_ASPACE_IMPL_H_

namespace __impl { namespace hal { namespace detail {

namespace virt {

inline
aspace::aspace(set_processing_unit &compatibleUnits, phys::aspace &pas, hal::error &err) :
    pUnits_(compatibleUnits),
    pas_(pas)
{
    ASSERTION(compatibleUnits.size() > 0);
    err = hal::error::HAL_SUCCESS;
}

inline
aspace::~aspace()
{
}

inline
const aspace::set_processing_unit &
aspace::get_processing_units() const
{
    return pUnits_;
}

inline
phys::aspace &
aspace::get_paspace()
{
    return pas_;
}

inline
const phys::aspace &
aspace::get_paspace() const
{
    return pas_;
}

} // namespace virt

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
