#ifndef GMAC_HAL_DETAIL_PHYS_OBJECT_H_
#define GMAC_HAL_DETAIL_PHYS_OBJECT_H_

#include <set>

#include "util/factory.h"
#include "util/misc.h"
#include "util/unique.h"

#include "hal/detail/ptr.h"

namespace __impl { namespace hal { namespace detail {
    
namespace phys {
    class memory;
    class processing_unit;
}

namespace virt {

class aspace;
class object_view;

class GMAC_LOCAL object :
    public util::unique<object>,
    protected util::factory<object_view> {
    friend class util::factory<object>;

public:
    typedef std::map<aspace *, object_view *> map_view;
    typedef std::set<const object_view *> set_view;
private:
    const phys::memory *memory_;
    size_t size_;
    map_view views_;

    object(const phys::memory &location, size_t size);
    ~object()
    {
    }
public:
    object_view *create_view(aspace &as, ptr::offset_type offset, gmacError_t &err);
    gmacError_t destroy_view(object_view &view);

    const phys::memory &get_memory() const;

    size_t get_size() const
    {
        return size_;
    }

    gmacError_t migrate(phys::memory &newLocation);

    const object_view *get_view(aspace &as) const
    {
        map_view::const_iterator it = views_.find(&as);

        return it != views_.end()? it->second: NULL;
    } 

    set_view get_views() const;

    set_view get_views(detail::phys::processing_unit::type type) const;
    set_view get_views(detail::phys::aspace &as) const;
};

typedef util::shared_ptr<object>       object_ptr;
typedef util::shared_ptr<const object> object_const_ptr;

class GMAC_LOCAL object_view :
    public util::unique<object_view> {
    friend class util::factory<object_view>;
private:
    object &parent_;
    aspace &aspace_;
    ptr::offset_type offset_;

    object_view(object &parent, aspace &as, ptr::offset_type offset) :
        parent_(parent),
        aspace_(as),
        offset_(offset)
    {
    }

    ~object_view()
    {
    }

public:
    object &get_object()
    {
        return parent_;
    }

    const object &get_object() const
    {
        return parent_;
    }

    aspace &get_vaspace()
    {
        return aspace_;
    }

    const aspace &get_vaspace() const
    {
        return aspace_;
    }

    ptr::offset_type get_offset() const
    {
        return offset_;
    }
};

}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
