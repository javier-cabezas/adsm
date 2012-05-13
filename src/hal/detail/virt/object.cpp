#include "hal/detail/types.h"

#include "object.h"

namespace __impl { namespace hal { namespace detail {
    
namespace virt {

object::object(const phys::memory &location, size_t size) :
    memory_(&location),
    size_(size)
{
}

object_view *
object::create_view(aspace &as, ptr::offset_type offset, hal::error &err)
{
    ASSERTION(as.get_paspace().get_memories().find(memory_) !=
              as.get_paspace().get_memories().end());
    // TODO: check if this is already working
#if 0
    // TODO: add support for multiple views of the same object in an Address Space
    if (views_.find(&as) != views_.end()) {
        err = HAL_ERROR_FEATURE_NOT_SUPPORTED;
        return NULL;
    }
#endif

    object_view *ret = create(*this, as, offset);

    map_view::iterator it = views_.find(&as);
    if (it != views_.end()) {
        it->second.insert(ret);
    } else {
        set_view viewsAspace;
        viewsAspace.insert(ret);

        views_.insert(map_view::value_type(&as, viewsAspace));
    }

    TRACE(LOCAL, FMT_ID2" insert " FMT_ID2":" FMT_ID2, get_print_id2(), as.get_print_id2(), ret->get_print_id2());

    err = HAL_SUCCESS;

    return ret;
}

hal::error
object::destroy_view(object_view &view)
{
    TRACE(LOCAL, FMT_ID2" destroy " FMT_ID2":" FMT_ID2, get_print_id2(), view.get_vaspace().get_print_id2(), view.get_print_id2());

    map_view::iterator it = views_.find(&view.get_vaspace());
    if (it == views_.end()) return HAL_ERROR_INVALID_VALUE;

    // Remove from the mapping list
    it->second.erase(&view);
    // Destroy object
    destroy(view);

    return HAL_SUCCESS;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
