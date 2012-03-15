#ifndef GMAC_HAL_DETAIL_PHYS_OBJECT_IMPL_H_
#define GMAC_HAL_DETAIL_PHYS_OBJECT_IMPL_H_

namespace __impl { namespace hal { namespace detail {
    
namespace virt {

object::object(phys::memory &location, size_t size) :
    memory_(location),
    size_(size)
{
}

gmacError_t
object::add_view(object_view &view)
{
    ASSERTION(view.get_vaspace().get_paspace().get_memories().find(memory_) !=
              view.get_vaspace().get_paspace().get_memories().end());

    views_.insert(map_view::value_type(&view.get_vaspace(), &view));
}

const phys::memory &
get_memory() const
{
    return memory_;
}

}

}}}

#endif /* OBJECT_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
