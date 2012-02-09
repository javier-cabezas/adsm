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
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
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

    inline
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

    inline
    _common_ptr_t(host_ptr ptr) :
        ptrHost_(ptr)
    {
    }

    inline
    _common_ptr_t(host_const_ptr ptr) :
        ptrHost_(ptr)
    {
    }
};

template <bool Const, typename Ptr, typename Ctx>
class GMAC_LOCAL _base_ptr_t :
    public _common_ptr_t<Const> {

    typedef _common_ptr_t<Const> parent;

    friend class _base_ptr_t<false, Ptr, Ctx>;
    friend class _base_ptr_t<true,  Ptr, Ctx>;

protected:
    Ptr ptrDev_;
    Ctx *ctx_;

public:
    static const char *address_fmt;
    static const char *offset_fmt;

    typedef typename parent::address_type address_type;
    typedef size_t offset_type;

    typedef Ptr backend_ptr;
    typedef typename Ptr::backend_type backend_type;

    inline
    bool is_const() const
    {
        return Const;
    }

    inline
    _base_ptr_t() :
        parent(0),
        ptrDev_(0),
        ctx_(0)
    {
    }

    inline
    _base_ptr_t(Ptr ptr, Ctx *ctx) :
        parent(0),
        ptrDev_(ptr),
        ctx_(ctx)
    {
    }

    inline
    explicit _base_ptr_t(host_ptr ptr) :
        parent(ptr),
        ptrDev_(0),
        ctx_(0)
    {
    }

    template <bool Const2 = Const>
    inline
    explicit _base_ptr_t(typename std::enable_if<Const2, host_const_ptr>::type ptr) :
        parent(ptr),
        ptrDev_(0),
        ctx_(0)
    {
    }


    inline
    _base_ptr_t(const _base_ptr_t &ptr) :
        parent(ptr),
        ptrDev_(ptr.ptrDev_),
        ctx_(ptr.ctx_)
    {
    }

    template <bool Const2>
    inline
    _base_ptr_t(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) :
        parent(ptr.ptrHost_),
        ptrDev_(ptr.ptrDev_),
        ctx_(ptr.ctx_)
    {
    }

    inline
    operator bool() const
    {
        return (this->get_context() != NULL && ptrDev_ != NULL) || this->ptrHost_ != NULL;
    }

    template <bool Const2>
    inline
    _base_ptr_t &
    operator=(const typename std::enable_if<!Const && Const2, _base_ptr_t<Const2, Ptr, Ctx> >::type &ptr)
    {
        if (this != &ptr) {
            this->ptrHost_ = ptr.ptrHost_;
            ptrDev_ = ptr.ptrDev_;
            ctx_    = ptr.ctx_;
        }
        return *this;
    }

    template <bool Const2>
    inline
    bool
    operator==(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        ASSERTION(ptr.ctx_ == this->ctx_, "Comparing pointers from different address spaces");
        bool ret;
        if (ctx_ == NULL) {
            ret = (this->ptrHost_ == ptr.ptrHost_);
        } else {
            ret = (ptrDev_ == ptr.ptrDev_);
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator!=(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        return !(*this == ptr);
    }

    template <bool Const2>
    inline
    bool
    operator<(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        ASSERTION(ptr.ctx_ == this->ctx_, "Comparing pointers from different address spaces");

        bool ret;
        if (ctx_ == NULL) {
            return this->ptrHost_ < ptr.ptrHost_;
        } else {
            return ptrDev_  < ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator<=(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        ASSERTION(ptr.ctx_ == this->ctx_, "Comparing pointers from different address spaces");

        bool ret;
        if (ctx_ == NULL) {
            return this->ptrHost_ <= ptr.ptrHost_;
        } else {
            return ptrDev_  <= ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator>(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        ASSERTION(ptr.ctx_ == this->ctx_, "Comparing pointers from different address spaces");

        bool ret;
        if (ctx_ == NULL) {
            return this->ptrHost_ > ptr.ptrHost_;
        } else {
            return ptrDev_  > ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator>=(const _base_ptr_t<Const2, Ptr, Ctx> &ptr) const
    {
        ASSERTION(ptr.ctx_ == this->ctx_, "Comparing pointers from different address spaces");

        bool ret;
        if (ctx_ == NULL) {
            return this->ptrHost_ >= ptr.ptrHost_;
        } else {
            return ptrDev_  >= ptr.ptrDev_;
        }
        return ret;

    }

    template <typename T>
    inline
    _base_ptr_t &
    operator+=(const T &off)
    {
        if (ctx_ == NULL) {
            this->ptrHost_ += off;
        } else {
            ptrDev_ += off;
        }
        return *this;
    }

    template <typename T>
    inline
    const _base_ptr_t
    operator+(const T &off) const
    {
        _base_ptr_t ret(*this);
        ret += off;
        return ret;
    }

    template <typename T>
    inline
    _base_ptr_t &
    operator-=(const T &off)
    {
        if (ctx_ == NULL) {
            ASSERTION(this->ptrHost_ >= host_ptr(off));
            this->ptrHost_ -= off;
        } else {
            ptrDev_ -= off;
        }
        return *this;
    }

    template <typename T>
    inline
    const _base_ptr_t
    operator-(const T &off) const
    {
        _base_ptr_t ret(*this);
        ret -= off;
        return ret;
    }

    inline
    typename Ptr::backend_type
    get_base() const
    {
        ASSERTION(is_device_ptr());
        return ptrDev_.get_base();
    }

    template <bool Const2 = Const>
    inline
    typename std::enable_if<Const2, host_const_ptr>::type
    get_host_addr() const
    {
        ASSERTION(is_host_ptr());
        return this->ptrHost_;
    }

    template <bool Const2 = Const>
    inline
    typename std::enable_if<!Const2, host_ptr>::type
    get_host_addr() const
    {
        ASSERTION(is_host_ptr());
        return this->ptrHost_;
    }

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

    inline
    size_t
    get_offset() const
    {
        ASSERTION(is_device_ptr());
        return ptrDev_.offset();
    }

    template <bool Const2 = Const>
    inline
    typename std::enable_if<!Const2, Ctx>::type *
    get_context()
    {
        return ctx_;
    }

    inline
    const Ctx *
    get_context() const
    {
        return ctx_;
    }

    inline
    bool
    is_host_ptr() const
    {
        return this->ctx_ == NULL;
    }

    inline
    bool
    is_device_ptr() const
    {
        return this->ctx_ != NULL;
    }
};

template <bool Const, typename Ptr, typename Ctx>
const char *_base_ptr_t<Const, Ptr, Ctx>::address_fmt = "%p";

template <bool Const, typename Ptr, typename Ctx>
const char *_base_ptr_t<Const, Ptr, Ctx>::offset_fmt = FMT_SIZE;


template <typename Ptr, typename Ctx>
class GMAC_LOCAL _const_ptr_t :
    public _base_ptr_t<true, Ptr, Ctx> {
    typedef _base_ptr_t<true, Ptr, Ctx> parent;
    typedef _base_ptr_t<false, Ptr, Ctx> parent_noconst;

public:
    inline
    _const_ptr_t() :
        parent()
    {
    }

    inline
    _const_ptr_t(Ptr ptr, Ctx *ctx) :
        parent(ptr, ctx)
    {
    }

    inline
    explicit _const_ptr_t(host_const_ptr ptr) :
        parent(ptr)
    {
    }

    inline
    explicit _const_ptr_t(host_ptr ptr) :
        parent(ptr)
    {
    }

    inline
    _const_ptr_t(const parent &ptr) :
        parent(ptr)
    {
    }

    inline
    _const_ptr_t(const parent_noconst &ptr) :
        parent(ptr)
    {
    }
};

template <typename Ptr, typename Ctx>
class GMAC_LOCAL _ptr_t :
    public _base_ptr_t<false, Ptr, Ctx> {
    typedef _base_ptr_t<false, Ptr, Ctx> parent;

public:
    inline
    _ptr_t() :
        parent()
    {
    }

    inline
    _ptr_t(Ptr ptr, Ctx *ctx) :
        parent(ptr, ctx)
    {
    }

    inline
    explicit _ptr_t(host_ptr ptr) :
        parent(ptr)
    {
    }

    inline
    _ptr_t(const _ptr_t &ptr) :
        parent(ptr)
    {
    }

    inline
    _ptr_t(const parent &ptr) :
        parent(ptr)
    {
    }
};

}}

#include "ptr-impl.h"

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
