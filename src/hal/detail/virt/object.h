#ifndef GMAC_HAL_DETAIL_PHYS_OBJECT_H_
#define GMAC_HAL_DETAIL_PHYS_OBJECT_H_

#include <set>

#include "util/unique.h"

namespace __impl { namespace hal { namespace detail {
    
namespace virt {

class aspace;
class object_view;

class GMAC_LOCAL object :
    public util::unique<object> {
public:
    typedef std::map<aspace *, object_view *> map_view;
private:
    phys::memory *memory_;
    size_t size_;
    map_view views_;

public:
    object(phys::memory &location, size_t size);

    object_view *create_view(aspace &as, ptrdiff_t offset, gmacError_t err);

    const phys::memory &get_memory() const;

    size_t get_size() const
    {
        return size_;
    }

    gmacError_t migrate(phys::memory &newLocation);
};

typedef util::shared_ptr<object>       object_ptr;
typedef util::shared_ptr<const object> object_const_ptr;

class GMAC_LOCAL object_view {
private:
    friend class object;

    object &parent_;
    aspace &aspace_;
    ptrdiff_t &offset_;

    object_view(object &parent, aspace &as, ptrdiff_t offset) :
        parent_(parent),
        aspace_(as),
        offset_(offset)
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

    ptrdiff_t get_offset() const
    {
        return offset_;
    }
};

}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
