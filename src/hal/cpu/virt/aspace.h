#ifndef GMAC_HAL_CPU_VIRT_ASPACE_H_
#define GMAC_HAL_CPU_VIRT_ASPACE_H_

#include "hal/cpu/types.h"

namespace __impl { namespace hal {
    
namespace detail {
namespace code {
    class repository;
    class repository_view;
}
}
    
namespace cpu { namespace virt {

typedef hal::detail::virt::aspace hal_aspace;
typedef hal::detail::virt::object hal_object;

typedef hal::detail::event_ptr hal_event_ptr;

class GMAC_LOCAL aspace :
    public hal_aspace {
        
    typedef hal_aspace parent;

public:
    aspace(hal_aspace::set_processing_unit &compatibleUnits, phys::aspace &pas, gmacError_t &err);

public:
    const set_processing_unit &get_processing_units() const;

    phys::aspace &get_paspace();
    const phys::aspace &get_paspace() const;

    ptr map(hal_object &obj, GmacProtection prot, gmacError_t &err);
    ptr map(hal_object &obj, GmacProtection prot, ptrdiff_t offset, gmacError_t &err);

    detail::code::repository_view *map(const detail::code::repository &repo, gmacError_t &err);

    gmacError_t unmap(ptr p);
    gmacError_t unmap(detail::code::repository_view &view);

    bool has_direct_copy(hal::const_ptr ptr1, hal::const_ptr ptr2);

    hal_event_ptr copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, gmacError_t &err);

    hal_event_ptr copy(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr memset(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr copy_async(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, gmacError_t &err);
    hal_event_ptr memset_async(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, gmacError_t &err);
};

}}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
