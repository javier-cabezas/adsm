#ifndef GMAC_HAL_DETAIL_PHYS_MEMORY_H_
#define GMAC_HAL_DETAIL_PHYS_MEMORY_H_

#include <set>

#include "util/unique.h"

namespace __impl { namespace hal { namespace detail { namespace phys {

class processing_unit;

class GMAC_LOCAL memory :
    public util::unique<memory> {
    size_t size_;
    typedef std::set<processing_unit *> set_processing_unit;
    set_processing_unit attachedUnits_;

    friend class processing_unit;

    void add_attached_unit(processing_unit &pu)
    {
        attachedUnits_.insert(&pu);
    }

public:
    memory(size_t size) :
        size_(size)
    {
        TRACE(LOCAL, FMT_ID2" Create", get_print_id2());
    }

    ~memory()
    {
        TRACE(LOCAL, FMT_ID2" Destroy", get_print_id2());
    }

    const set_processing_unit &get_attached_units() const
    {
        return attachedUnits_;
    }

    size_t get_size() const
    {
        return size_;
    }

    //bool is_host_memory() const;
};

}}}}

#include "memory-impl.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
