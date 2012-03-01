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

template <bool C, typename T, typename T2>
struct static_union {
    typedef T type;
};

template <typename T, typename T2>
struct static_union<false, T, T2> {
    typedef T2 type;
};

template <bool Const, typename Ptr, typename Aspace, typename D>
class GMAC_LOCAL _base_ptr_t {
    friend class _base_ptr_t<false, Ptr, Aspace, D>;
    friend class _base_ptr_t<true,  Ptr, Aspace, D>;

protected:
    typedef typename static_union<Const, host_const_ptr, host_ptr>::type HPtr;

    Aspace *aspace_;

    HPtr ptrHost_;
    Ptr ptrDev_;

public:
    static const char *address_fmt;
    static const char *offset_fmt;

    typedef HPtr address_type;
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
        aspace_(0),
        ptrHost_(0),
        ptrDev_(0)
    {
    }

    inline
    _base_ptr_t(Ptr ptr, Aspace *aspace) :
        aspace_(aspace),
        ptrHost_(0),
        ptrDev_(ptr)
    {
        ASSERTION(aspace != NULL);
    }

    inline
    explicit _base_ptr_t(host_ptr ptr, Aspace *aspace) :
        aspace_(aspace),
        ptrHost_(ptr),
        ptrDev_(0)
    {
        ASSERTION(aspace != NULL);
    }

    template <bool Const2 = Const>
    inline
    explicit _base_ptr_t(typename std::enable_if<Const2, host_const_ptr>::type ptr, Aspace *aspace) :
        aspace_(aspace),
        ptrHost_(ptr),
        ptrDev_(0)
    {
        ASSERTION(aspace != NULL);
    }

    inline
    _base_ptr_t(const _base_ptr_t &ptr) :
        aspace_(ptr.aspace_),
        ptrHost_(ptr.ptrHost_),
        ptrDev_(ptr.ptrDev_)
    {
    }

    template <bool Const2>
    inline
    _base_ptr_t(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) :
        aspace_(ptr.aspace_),
        ptrHost_(ptr.ptrHost_),
        ptrDev_(ptr.ptrDev_)
    {
    }

    inline
    operator bool() const
    {
        return (this->get_aspace() != NULL);
    }

    template <bool Const2>
    inline
    _base_ptr_t &
    operator=(const typename std::enable_if<!Const && Const2, _base_ptr_t<Const2, Ptr, Aspace, D> >::type &ptr)
    {
        if (this != &ptr) {
            ptrHost_ = ptr.ptrHost_;
            ptrDev_ = ptr.ptrDev_;
            aspace_ = ptr.aspace_;
        }
        return *this;
    }

    template <bool Const2>
    inline
    bool
    operator==(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        ASSERTION(ptr.aspace_ == this->aspace_, "Comparing pointers from different address spaces");
        bool ret;
        if (is_host_ptr()) {
            ret = (this->ptrHost_ == ptr.ptrHost_);
        } else {
            ret = (ptrDev_ == ptr.ptrDev_);
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator!=(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        return !(*this == ptr);
    }

    template <bool Const2>
    inline
    bool
    operator<(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        ASSERTION(ptr.aspace_ == this->aspace_, "Comparing pointers from different address spaces");

        bool ret;
        if (is_host_ptr()) {
            return this->ptrHost_ < ptr.ptrHost_;
        } else {
            return ptrDev_  < ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator<=(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        ASSERTION(ptr.aspace_ == this->aspace_, "Comparing pointers from different address spaces");

        bool ret;
        if (is_host_ptr()) {
            return this->ptrHost_ <= ptr.ptrHost_;
        } else {
            return ptrDev_  <= ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator>(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        ASSERTION(ptr.aspace_ == this->aspace_, "Comparing pointers from different address spaces");

        bool ret;
        if (is_host_ptr()) {
            return this->ptrHost_ > ptr.ptrHost_;
        } else {
            return ptrDev_  > ptr.ptrDev_;
        }
        return ret;
    }

    template <bool Const2>
    inline
    bool
    operator>=(const _base_ptr_t<Const2, Ptr, Aspace, D> &ptr) const
    {
        ASSERTION(ptr.aspace_ == this->aspace_, "Comparing pointers from different address spaces");

        bool ret;
        if (is_host_ptr()) {
            return this->ptrHost_ >= ptr.ptrHost_;
        } else {
            return this->ptrDev_  >= ptr.ptrDev_;
        }
        return ret;

    }

    template <typename T>
    inline
    _base_ptr_t &
    operator+=(const T &off)
    {
        if (is_host_ptr()) {
            this->ptrHost_ += off;
        } else {
            this->ptrDev_ += off;
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
    operator++() const
    {
        *this += 1;
        return *this;
    }

    template <typename T>
    inline
    _base_ptr_t
    operator++(int dummy) const
    {
        _base_ptr_t ret(*this);
        *this += 1;
        return ret;
    }

    template <typename T>
    inline
    _base_ptr_t &
    operator-=(const T &off)
    {
        if (is_host_ptr()) {
            ASSERTION(this->ptrHost_ >= host_ptr(long_t(off)));
            this->ptrHost_ -= off;
        } else {
            this->ptrDev_ -= off;
        }
        return *this;
    }

#if 0
    template <>
    _base_ptr_t &
    operator-=(const _base_ptr_t &ptr)
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

    template <typename T>
    inline
    const _base_ptr_t
    operator-(const T &off) const
    {
        _base_ptr_t ret(*this);
        ret -= off;
        return ret;
    }

    template <typename T>
    inline
    const _base_ptr_t
    operator-(const _base_ptr_t &ptr) const
    {
        ASSERTION(aspace_ == ptr.aspace_);
        _base_ptr_t ret(*this);

        if (is_host_ptr()) {
            ret.ptrHost_ -= ptr.ptrHost_;
        } else {
            ASSERTION(get_base()   == ptr.get_base());
            ASSERTION(get_offset() >  ptr.get_offset());
            ret.ptrDev_ -= ptr.get_offset();
        }

        return ret;
    }

    template <typename T>
    inline
    _base_ptr_t &
    operator--() const
    {
        *this -= 1;
        return *this;
    }

    template <typename T>
    inline
    _base_ptr_t
    operator--(int dummy) const
    {
        _base_ptr_t ret(*this);
        *this -= 1;
        return ret;
    }

    inline
    typename Ptr::backend_type
    get_base() const
    {
        ASSERTION(is_device_ptr());
        return this->ptrDev_.get_base();
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
        return this->ptrDev_.offset();
    }

#if 0
    template <bool Const2 = Const>
    inline
    typename std::enable_if<!Const2, Aspace>::type *
    get_aspace()
    {
        return aspace_;
    }
#endif

    inline
    Aspace *
    get_aspace()
    {
        return aspace_;
    }

    inline
    const Aspace *
    get_aspace() const
    {
        return aspace_;
    }

    inline
    bool
    is_host_ptr() const
    {
        D &d = this->aspace_->get_device();
        return d.get_type() == D::DEVICE_TYPE_CPU;
    }

    inline
    bool
    is_device_ptr() const
    {
        D &d = this->aspace_->get_device();
        return d.get_type() != D::DEVICE_TYPE_CPU;
    }
};

#if 0
template <>
template <>
_base_ptr_t<true, Ptr, Aspace, D> &
_base_ptr_t<true, Ptr, Aspace, D>::operator-=(const _base_ptr_t<Const, Ptr, Aspace, D> &ptr)
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

template <bool Const, typename Ptr, typename Aspace, typename D>
const char *_base_ptr_t<Const, Ptr, Aspace, D>::address_fmt = "%p";

template <bool Const, typename Ptr, typename Aspace, typename D>
const char *_base_ptr_t<Const, Ptr, Aspace, D>::offset_fmt = FMT_SIZE;


template <typename Ptr, typename Aspace, typename D>
class GMAC_LOCAL _const_ptr_t :
    public _base_ptr_t<true, Ptr, Aspace, D> {
    typedef _base_ptr_t<true, Ptr, Aspace, D> parent;
    typedef _base_ptr_t<false, Ptr, Aspace, D> parent_noconst;

public:
    inline
    _const_ptr_t() :
        parent()
    {
    }

    inline
    _const_ptr_t(Ptr ptr, Aspace *aspace) :
        parent(ptr, aspace)
    {
    }

    inline
    explicit _const_ptr_t(host_const_ptr ptr, Aspace *aspace) :
        parent(ptr, aspace)
    {
    }

    inline
    explicit _const_ptr_t(host_ptr ptr, Aspace *aspace) :
        parent(ptr, aspace)
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

template <typename Ptr, typename Aspace, typename D>
class GMAC_LOCAL _ptr_t :
    public _base_ptr_t<false, Ptr, Aspace, D> {
    typedef _base_ptr_t<false, Ptr, Aspace, D> parent;

public:
    inline
    _ptr_t() :
        parent()
    {
    }

    inline
    _ptr_t(Ptr ptr, Aspace *aspace) :
        parent(ptr, aspace)
    {
    }

    inline
    explicit _ptr_t(host_ptr ptr, Aspace *aspace) :
        parent(ptr, aspace)
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

    static _ptr_t null;
};


template <typename Ptr, typename Aspace, typename D>
_ptr_t<Ptr, Aspace, D> _ptr_t<Ptr, Aspace, D>::null = _ptr_t();

}}

#ifdef USE_CUDA
#include "cuda/ptr.h"
#else
#include "opencl/ptr.h"
#endif

namespace __impl { namespace hal { namespace detail {
class aspace;
class device;
}}}

namespace __impl { namespace hal {
typedef __impl::hal::_ptr_t<
                            _cuda_ptr_t,
                            detail::aspace,
                            detail::device
                            > ptr;

typedef __impl::hal::_const_ptr_t<
                                   _cuda_ptr_t,
                                   detail::aspace,
                                   detail::device
                                 > const_ptr;
}}

//#include "ptr-impl.h"

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
