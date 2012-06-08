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
typedef hal::detail::virt::context hal_context;
typedef hal::detail::virt::object hal_object;

typedef hal::detail::event_ptr hal_event_ptr;

class GMAC_LOCAL aspace :
    public hal_aspace {
        
    typedef hal_aspace parent;

public:
    aspace(hal_aspace::set_processing_unit &compatibleUnits, phys::aspace &pas, hal::error &err);

public:
    const set_processing_unit &get_processing_units() const;

    phys::aspace &get_paspace();
    const phys::aspace &get_paspace() const;

    ptr map(hal_object &obj, GmacProtection prot, hal::error &err);
    ptr map(hal_object &obj, GmacProtection prot, ptrdiff_t offset, hal::error &err);

    detail::code::repository_view *map(const detail::code::repository &repo, hal::error &err);

    hal::error unmap(ptr p);
    hal::error unmap(detail::code::repository_view &view);

    hal::error protect(hal::ptr obj, size_t count, GmacProtection prot);

    bool has_direct_copy(hal::const_ptr ptr1, hal::const_ptr ptr2);

    hal_event_ptr copy(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err);

    hal_event_ptr copy(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr copy(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr memset(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr copy_async(hal::ptr dst, device_input &input, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr copy_async(device_output &output, hal::const_ptr src, size_t count, list_event_detail *dependencies, hal::error &err);
    hal_event_ptr memset_async(hal::ptr dst, int c, size_t count, list_event_detail *dependencies, hal::error &err);

    hal_context *create_context(hal::error &err);
    void destroy_context(hal_context &ctx);
};

}}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
