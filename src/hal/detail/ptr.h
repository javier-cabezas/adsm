/* Copyright (c) 2009-2011 University of Illinois
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

template <bool Const>
class GMAC_LOCAL _common_ptr_t {
protected:
    typedef host_ptr address_type;

    host_ptr ptrHost_;

    _common_ptr_t(host_ptr ptr) :
        ptrHost_(ptr)
    {
    }
};

template <>
class GMAC_LOCAL _common_ptr_t<true> {
protected:
    typedef host_const_ptr address_type;

    host_const_ptr ptrHost_;

    _common_ptr_t(host_ptr ptr) :
        ptrHost_(ptr)
    {
    }

    _common_ptr_t(host_const_ptr ptr) :
        ptrHost_(ptr)
    {
    }
};

template <bool C, typename T, typename T2>
struct static_union {
    typedef T type;
};

template <typename T, typename T2>
struct static_union<false, T, T2> {
    typedef T2 type;
};

template <typename Base>
class GMAC_LOCAL backend_ptr_t
{
protected:
    Base base_;
    backend_ptr_t(Base base) :
        base_(base)
    {
    }

public:
    typedef Base backend_type;

    virtual backend_ptr_t &
    operator=(const backend_ptr_t &ptr) = 0;

    virtual bool
    operator==(const backend_ptr_t &ptr) const = 0;

    virtual bool
    operator!=(const backend_ptr_t &ptr) const = 0;

    virtual bool
    operator< (const backend_ptr_t &ptr) const = 0;

    virtual bool
    operator<=(const backend_ptr_t &ptr) const = 0;

    virtual bool
    operator> (const backend_ptr_t &ptr) const = 0;

    virtual bool
    operator>=(const backend_ptr_t &ptr) const = 0;

    virtual backend_ptr_t &
    operator+=(const ptrdiff_t &off) = 0;

    virtual backend_ptr_t &
    operator-=(const ptrdiff_t &off) = 0;

    backend_type
    get_base() const
    {
        return base_;
    }

    virtual void *
    get() const = 0;

    virtual ptrdiff_t
    offset() const = 0;
};

template <bool Const, typename View>
class GMAC_LOCAL _base_ptr_t {
    friend class _base_ptr_t<false, View>;
    friend class _base_ptr_t<true,  View>;

protected:
#if 0
    typedef typename static_union<Const,          // Condition
                                  host_const_ptr, // If true
                                  host_ptr        // If false
                                  >::type HPtr;
#endif
    View *view_;
    ptrdiff_t offset_;

    //HPtr ptrHost_;
    //Ptr ptrDev_;

public:
    static const char *address_fmt;
    static const char *offset_fmt;

    //typedef HPtr address_type;
    typedef ptrdiff_t offset_type;

    //typedef Ptr backend_ptr;
    //typedef typename Ptr::backend_type backend_type;

    bool is_const() const
    {
        return Const;
    }

    _base_ptr_t() :
        view_(NULL),
        offset_(0)
    {
    }

    // TODO check if view has to be a reference
    explicit _base_ptr_t(View &view, ptrdiff_t offset = 0) :
        view_(&view),
        offset_(offset)
    {
        ASSERTION(view_ != NULL);
    }

#if 0
    template <bool Const2 = Const>
    explicit _base_ptr_t(typename std::enable_if<Const2, host_const_ptr>::type ptr, View *view) :
        view_(view),
        ptrHost_(ptr),
        ptrDev_(0)
    {
        ASSERTION(aspace != NULL);
    }
#endif

    _base_ptr_t(const _base_ptr_t &ptr) :
        view_(ptr.view_),
        offset_(ptr.offset_)
    {
    }

    template <bool Const2>
    _base_ptr_t(const _base_ptr_t<Const2, View> &ptr) :
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
    _base_ptr_t &
    operator=(const _base_ptr_t<Const2, View> &ptr)
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
    operator==(const _base_ptr_t<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == view_, "Comparing pointers from different address spaces");

        return (ptr.view_ == view_) && (offset_ == offset_);
    }

    template <bool Const2>
    bool
    operator!=(const _base_ptr_t<Const2, View> &ptr) const
    {
        return !(*this == ptr);
    }

    template <bool Const2>
    bool
    operator<(const _base_ptr_t<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ < ptr.offset_;
    }

    template <bool Const2>
    bool
    operator<=(const _base_ptr_t<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ <= ptr.offset_;
    }

    template <bool Const2>
    bool
    operator>(const _base_ptr_t<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ > ptr.offset_;
    }

    template <bool Const2>
    bool
    operator>=(const _base_ptr_t<Const2, View> &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Comparing pointers from different views");

        return offset_ >= ptr.offset_;
    }

    template <typename T>
    _base_ptr_t &
    operator+=(const T &off)
    {
        //ASSERTION(offset_ + off < view_->get_object().get_size(), "Out of view boundaries");
        
        offset_ += off;

        return *this;
    }

    template <typename T>
    const _base_ptr_t
    operator+(const T &off) const
    {
        _base_ptr_t ret(*this);

        // operator+= performs the check for view boundaries
        ret += off;

        return ret;
    }

    template <typename T>
    _base_ptr_t &
    operator++() const
    {
        // operator+= performs the check for view boundaries
        *this += 1;
        return *this;
    }

    template <typename T>
    _base_ptr_t
    operator++(int dummy) const
    {
        _base_ptr_t ret(*this);
        // operator+= performs the check for view boundaries
        *this += 1;
        return ret;
    }

    template <typename T>
    _base_ptr_t &
    operator-=(const T &off)
    {
        if (off > 0) {
            ASSERTION(offset_ >= ptrdiff_t(off), "Out of view boundaries");
        } else if (off < 0) {
            //ASSERTION(offset_ - off < view_->get_object().get_size(), "Out of view boundaries");
        }

        offset_ -= off;

        return *this;
    }

    template <typename T>
    const _base_ptr_t
    operator-(const T &off) const
    {
        _base_ptr_t ret(*this);

        // operator-= performs the check for view boundaries
        ret -= off;
        return ret;
    }

    template <typename T>
    ptrdiff_t
    operator-(const _base_ptr_t &ptr) const
    {
        ASSERTION(ptr.view_ == this->view_, "Subtracting pointers from different views");

        return offset_ - ptr.offset_;
    }

    template <typename T>
    _base_ptr_t &
    operator--() const
    {
        // operator-= performs the check for view boundaries
        *this -= 1;
        return *this;
    }

    template <typename T>
    _base_ptr_t
    operator--(int dummy) const
    {
        _base_ptr_t ret(*this);
        // operator-= performs the check for view boundaries
        *this -= 1;
        return ret;
    }

#if 0
    typename Ptr::backend_type
    get_base() const
    {
        ASSERTION(is_device_ptr());
        return this->ptrDev_.get_base();
    }
#endif
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

#if 0
    template <bool Const2 = Const>
    typename std::enable_if<Const2, host_const_ptr>::type
    get_host_addr() const
    {
        ASSERTION(is_host_ptr());
        return this->ptrHost_;
    }

    template <bool Const2 = Const>
    typename std::enable_if<!Const2, host_ptr>::type
    get_host_addr() const
    {
        ASSERTION(is_host_ptr());
        return this->ptrHost_;
    }
#endif

#if 0
    host_ptr_t
    get_addr() const
    {
        if (is_host_ptr()) {
            return ptrHost_;
        } else {
            return HPtr(ptrDev_.get()) + ptrDev_.offset();
        }
    }
#endif

    ptrdiff_t
    get_offset() const
    {
        return offset_;
    }

#if 0
    template <bool Const2 = Const>
    typename std::enable_if<!Const2, Aspace>::type *
    get_aspace()
    {
        return aspace_;
    }

    decltype(view_->get_vaspace()) &
    get_aspace()
    {
        ASSERTION(view_);
        return &view_->get_vaspace();
    }

    const decltype(view_->get_vaspace()) &
    get_aspace() const
    {
        ASSERTION(view_);
        return &view_->get_vaspace();
    }
#endif


#if 0
    const Aspace *
    get_aspace() const
    {
        return aspace_;
    }

    bool
    is_host_ptr() const
    {
        auto &o = view_->get_object();
        return o.get_type() == D::PUNIT_TYPE_CPU;
    }

    bool
    is_device_ptr() const
    {
        D &d = this->aspace_->get_processing_unit();
        return d.get_type() != D::PUNIT_TYPE_CPU;
    }
#endif
};

#if 0
template <>
template <>
_base_ptr_t<true, Ptr, Aspace> &
_base_ptr_t<true, Ptr, Aspace>::operator-=(const _base_ptr_t<Const, Ptr, Aspace> &ptr)
{
    ASSERTION(aspace_ == ptr.aspace_);
    if (is_host_ptr()) {
        ptrHost_ -= ptr.ptrHost_;
    } else {
        ASSERTION(get_base()   == ptr.get_base());
        ASSERTION(get_offset() >  ptr.get_offset());
        ptrDev_ -= ptr.get_offset();
    }

    return *this;
}
#endif

template <bool Const, typename View>
const char *_base_ptr_t<Const, View>::address_fmt = "%p";

template <bool Const, typename View>
const char *_base_ptr_t<Const, View>::offset_fmt = FMT_SIZE;


template <typename View>
class GMAC_LOCAL _const_ptr_t :
    public _base_ptr_t<true, View> {
    typedef _base_ptr_t<true, View> parent;
    typedef _base_ptr_t<false, View> parent_noconst;

public:
    _const_ptr_t() :
        parent()
    {
    }

    explicit _const_ptr_t(View &view, ptrdiff_t offset = 0) :
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
    public _base_ptr_t<false, View> {
    typedef _base_ptr_t<false, View> parent;

public:
    _ptr_t() :
        parent()
    {
    }

    explicit _ptr_t(View &view, ptrdiff_t offset = 0) :
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

#if 0
#ifdef USE_CUDA
#include "hal/cuda/ptr.h"
#else
#include "hal/opencl/ptr.h"
#endif
#endif

//#include "virt/object.h"

namespace __impl { namespace hal {

namespace detail { namespace virt  {
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
