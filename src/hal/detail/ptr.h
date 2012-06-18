/* Copyright (c) 2009-2012 University of Illinois
                           Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, DNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef GMAC_HAL_PTR_H_
#define GMAC_HAL_PTR_H_

#include "util/smart_ptr.h"

namespace __impl { namespace hal {

template <bool Const, typename View>
class GMAC_LOCAL base_ptr {
    friend class base_ptr<false, View>;
    friend class base_ptr<true,  View>;

protected:
    View *view_;
    size_t offset_;

public:
    static const char *address_fmt;
    static const char *offset_fmt;

    typedef size_t offset_type;

    bool is_const() const
    {
        return Const;
    }

    base_ptr() :
        view_(NULL),
        offset_(0)
    {
    }

    // TODO check if view has to be a reference
    explicit base_ptr(View &view, offset_type offset = 0) :
        view_(&view),
        offset_(offset)
    {
        ASSERTION(view_ != NULL);
    }

    base_ptr(const base_ptr &ptr) :
        view_(ptr.view_),
        offset_(ptr.offset_)
    {
    }

    template <bool Const2>
    base_ptr(const base_ptr<Const2, View> &ptr) :
        view_(ptr.view_),
        offset_(ptr.offset_)
    {
        static_assert(Const || !Const2, "Cannot create non-const pointer from const pointer");
    }

    operator bool() const
    {
        return (view_ != NULL);
    }

    template <bool Const2>
    base_ptr &
    operator=(const base_ptr<Const2, View> &ptr)
    {
        static_assert(Const || !Const2, "Cannot assign const pointer to non-const pointer");

        if (this != &ptr) {
            view_   = ptr.view_;
            offset_ = ptr.offset_;
        }
        return *this;
    }

    template <bool Const2>
    bool
    operator==(const base_ptr<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == view_, "Comparing pointers from different address spaces");

        return (ptr.view_ == view_) && (offset_ == offset_);
    }

    template <bool Const2>
    bool
    operator!=(const base_ptr<Const2, View> &ptr) const
    {
        return !(*this == ptr);
    }

    template <bool Const2>
    bool
    operator<(const base_ptr<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ < ptr.offset_;
    }

    template <bool Const2>
    bool
    operator<=(const base_ptr<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ <= ptr.offset_;
    }

    template <bool Const2>
    bool
    operator>(const base_ptr<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ > ptr.offset_;
    }

    template <bool Const2>
    bool
    operator>=(const base_ptr<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ >= ptr.offset_;
    }

    template <typename T>
    base_ptr &
    operator+=(const T &off)
    {
        //ASSERTION(offset_ + off < view_->get_object().get_size(), "Out of view boundaries");
        
        offset_ += off;

        return *this;
    }

    template <typename T>
    const base_ptr
    operator+(const T &off) const
    {
        base_ptr ret(*this);

        // operator+= performs the check for view boundaries
        ret += off;

        return ret;
    }

    template <typename T>
    base_ptr &
    operator++() const
    {
        // operator+= performs the check for view boundaries
        *this += 1;
        return *this;
    }

    template <typename T>
    base_ptr
    operator++(int dummy) const
    {
        base_ptr ret(*this);
        // operator+= performs the check for view boundaries
        *this += 1;
        return ret;
    }

    template <typename T>
    base_ptr &
    operator-=(const T &off)
    {
        if (off > 0) {
            ASSERTION(offset_ >= offset_type(off), "Out of view boundaries");
        } else if (off < 0) {
            //ASSERTION(offset_ - off < view_->get_object().get_size(), "Out of view boundaries");
        }

        offset_ -= off;

        return *this;
    }

    template <typename T>
    const base_ptr
    operator-(const T &off) const
    {
        base_ptr ret(*this);

        // operator-= performs the check for view boundaries
        ret -= off;
        return ret;
    }

    template <typename T>
    ptrdiff_t
    operator-(const base_ptr &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Subtracting pointers from different views");

        return ptrdiff_t(offset_) - ptrdiff_t(ptr.offset_);
    }

    template <typename T>
    base_ptr &
    operator--() const
    {
        // operator-= performs the check for view boundaries
        *this -= 1;
        return *this;
    }

    template <typename T>
    base_ptr
    operator--(int dummy) const
    {
        base_ptr ret(*this);
        // operator-= performs the check for view boundaries
        *this -= 1;
        return ret;
    }

    View &
    get_view()
    {
        ASSERTION(view_);
        return *view_;
    }

    const View &
    get_view() const
    {
        ASSERTION(view_);
        return *view_;
    }

    offset_type
    get_offset() const
    {
        return offset_;
    }
};

template <bool Const, typename View>
const char *base_ptr<Const, View>::address_fmt = "%p";

template <bool Const, typename View>
const char *base_ptr<Const, View>::offset_fmt = FMT_SIZE;


template <typename View>
class GMAC_LOCAL _const_ptr_t :
    public base_ptr<true, View> {
    typedef base_ptr<true, View> parent;
    typedef base_ptr<false, View> parent_noconst;

public:
    typedef typename parent::offset_type offset_type;

    _const_ptr_t() :
        parent()
    {
    }

    explicit _const_ptr_t(View &view, offset_type offset = 0) :
        parent(view, offset)
    {
    }

    _const_ptr_t(const parent &ptr) :
        parent(ptr)
    {
    }

    _const_ptr_t(const parent_noconst &ptr) :
        parent(ptr)
    {
    }
};

template <typename View>
class GMAC_LOCAL _ptr_t :
    public base_ptr<false, View> {
    typedef base_ptr<false, View> parent;

public:
    typedef typename parent::offset_type offset_type;

    _ptr_t() :
        parent()
    {
    }

    explicit _ptr_t(View &view, offset_type offset = 0) :
        parent(view, offset)
    {
    }

    _ptr_t(const _ptr_t &ptr) :
        parent(ptr)
    {
    }

    _ptr_t(const parent &ptr) :
        parent(ptr)
    {
    }

    static _ptr_t null;
};

template <typename View>
_ptr_t<View> _ptr_t<View>::null = _ptr_t();

}}

namespace __impl { namespace hal {

namespace detail { namespace virt  {
    class object;
    class object_view;
}}

typedef __impl::hal::_ptr_t<
                               detail::virt::object_view
                           > ptr;

typedef __impl::hal::_const_ptr_t<
                                     detail::virt::object_view
                                 > const_ptr;

}}

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
