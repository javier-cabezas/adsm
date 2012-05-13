#ifndef GMAC_HAL_CPU_PHYS_ASPACE_H_
#define GMAC_HAL_CPU_PHYS_ASPACE_H_

namespace __impl { namespace hal { namespace cpu {

namespace virt {
class aspace;
}

namespace phys {

class processing_unit;

class GMAC_LOCAL aspace :
    public detail::phys::aspace {
    typedef detail::phys::aspace parent;
public:
    aspace(detail::phys::platform &plat, const set_memory &memories);
    virtual ~aspace();
    
    detail::virt::aspace *
    create_vaspace(detail::virt::aspace::set_processing_unit &compatibleUnits, hal::error &err);

    hal::error destroy_vaspace(detail::virt::aspace &as);
};

}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
