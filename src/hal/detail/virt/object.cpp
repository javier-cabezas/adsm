#include "hal/detail/types.h"

#include "object.h"

namespace __impl { namespace hal { namespace detail {
    
namespace virt {

object::object(phys::memory &location, size_t size) :
    memory_(&location),
    size_(size)
{
}

object_view *
object::create_view(aspace &as, ptr::offset_type offset, gmacError_t &err)
{
    ASSERTION(as.get_paspace().get_memories().find(memory_) !=
              as.get_paspace().get_memories().end());

    // TODO: add support for multiple views of the same object in an Address Space
    if (views_.find(&as) != views_.end()) {
        err = gmacErrorFeatureNotSupported;
        return NULL;
    }

    object_view *ret = create(*this, as, offset);
    views_.insert(map_view::value_type(&as, ret));

    TRACE(LOCAL, FMT_ID2" insert "FMT_ID2":"FMT_ID2, get_print_id2(), as.get_print_id2(), ret->get_print_id2());

    err = gmacSuccess;

    return ret;
}

gmacError_t
object::destroy_view(object_view &view)
{
    TRACE(LOCAL, FMT_ID2" destroy "FMT_ID2":"FMT_ID2, get_print_id2(), view.get_vaspace().get_print_id2(), view.get_print_id2());

    map_view::iterator it = views_.find(&view.get_vaspace());
    if (it == views_.end()) return gmacErrorInvalidValue;

    destroy(*it->second);
    views_.erase(it);

    return gmacSuccess;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
