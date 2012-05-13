#ifndef GMAC_TEST_UNIT_MOCK_HAL_PTR_H_
#define GMAC_TEST_UNIT_MOCK_HAL_PTR_H_

#include <stddef.h>

#include "include/gmac/visibility.h"

#include "virt.h"

namespace __impl { namespace hal {

class GMAC_LOCAL ptr {
    virt::object_view *view_;
    ptrdiff_t off_;

public:
    typedef ptrdiff_t offset_type;

    ptr() :
        view_(nullptr)
    {
    }

    ptr(virt::object_view &view, offset_type off) :
        view_(&view),
        off_(off)
    {
    }

    operator bool() const
    {
        return view_ != nullptr;
    }

    offset_type get_offset() const
    {
        return off_;
    }

    ptr &operator=(const ptr &p)
    {
        if (this != &p) {
            view_ = p.view_;
            off_  = p.off_;
        }
        return *this;
    }

    template <typename T>
    ptr operator+(T off) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ptr ret(*view_, off_ + offset_type(off));
        return ret;
    }

    template <typename T>
    ptr &operator+=(T off)
    {
        ASSERTION(bool(*this), "Invalid pointer");
        off_ += offset_type(off);
        return *this;
    }

    template <typename T>
    ptr operator-(T off) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ptr ret(*view_, off_ - offset_type(off));
        return ret;
    }

    template <typename T>
    ptr &operator-=(T off)
    {
        ASSERTION(bool(*this), "Invalid pointer");
        off_ -= offset_type(off);
        return *this;
    }

    bool operator<(const ptr &p) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ASSERTION(bool(p), "Invalid pointer");
        return off_ < p.off_;
    }

    bool operator<=(const ptr &p) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ASSERTION(bool(p), "Invalid pointer");
        return off_ <= p.off_;
    }

    bool operator>(const ptr &p) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ASSERTION(bool(p), "Invalid pointer");
        return off_ > p.off_;
    }

    bool operator>=(const ptr &p) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ASSERTION(bool(p), "Invalid pointer");
        return off_ >= p.off_;
    }

    bool operator==(const ptr &p) const
    {
        ASSERTION(bool(*this), "Invalid pointer");
        ASSERTION(bool(p), "Invalid pointer");
        return off_ == p.off_;
    }

    virt::object_view &get_view()
    {
        ASSERTION(bool(*this), "Invalid pointer");
        return *view_;
    }
};

typedef ptr const_ptr;

}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
