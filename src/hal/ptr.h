/* Copyright (c) 2009, 2010 University of Illinois
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

namespace __impl { namespace hal {

template <typename Ptr, typename C>
class GMAC_LOCAL _ptr_t {
protected:
    Ptr ptrDev_;
    hostptr_t ptrHost_;

    C *ctx_;

public:
    inline
    _ptr_t(Ptr ptr, C *ctx) :
        ptrDev_(ptr),
        ctx_(ctx)
    {
    }

    inline
    _ptr_t(typename Ptr::backend_type value, C *ctx) :
        ptrDev_(value),
        ctx_(ctx)
    {
    }

    inline
    explicit _ptr_t(hostptr_t ptr) :
        ptrDev_(0),
        ptrHost_(ptr),
        ctx_(NULL)
    {
    }

    inline
    _ptr_t() :
        ptrDev_(0),
        ptrHost_(0),
        ctx_(NULL)
    {
    }

    inline
    _ptr_t(const _ptr_t &ptr) :
        ptrDev_(ptr.ptrDev_),
        ptrHost_(ptr.ptrHost_),
        ctx_(ptr.ctx_)
    {
    }

    inline
    operator bool()
    {
        return ctx_ != NULL || ptrHost_ != NULL;
    }

    inline _ptr_t &
    operator=(const _ptr_t &ptr)
    {
        if (this != &ptr) {
            ptrDev_ = ptr.ptrDev_;
            ptrHost_ = ptr.ptrHost_;
            ctx_  = ptr.ctx_;
        }
        return *this;
    }

    inline bool
    operator==(const _ptr_t &ptr) const
    {
        bool ret;
        if (ctx_ == NULL) {
            ret = (ptrHost_ == ptr.ptrHost_);
        } else {
            ret = (ptrDev_ == ptr.ptrDev_);
        }
        return ret;
    }

    inline bool
    operator==(long i) const
    {
        bool ret;
        if (ctx_ == NULL) {
            ret = (ptrHost_ == hostptr_t(i));
        } else {
            ret = (ptrDev_ == i);
        }
        return ret;
    }

    inline bool
    operator!=(const _ptr_t &ptr) const
    {
        bool ret;
        if (ctx_ == NULL) {
            ret = (ptrHost_ != ptr.ptrHost_);
        } else {
            ret = (ptrDev_ != ptr.ptrDev_);
        }
        return ret;
    }

    inline bool
    operator!=(long i) const
    {
        bool ret;
        if (ctx_ == NULL) {
            ret = (ptrHost_ != hostptr_t(i));
        } else {
            ret = (ptrDev_ != i);
        }
        return ret;

    }

    inline bool
    operator<(const _ptr_t &ptr) const
    {
        bool ret;
        if (ctx_ == NULL) {
            return ptrHost_ < ptr.ptrHost_;
        } else {
            return ctx_ < ptr.ctx_ || (ctx_ == ptr.ctx_ && ptrDev_ < ptr.ptrDev_);
        }
        return ret;
    }

    template <typename T>
    inline _ptr_t &
    operator+=(const T &off)
    {
        if (ctx_ == NULL) {
            ptrHost_ += off;
        } else {
            ptrDev_ += off;
        }
        return *this;
    }

    template <typename T>
    inline const _ptr_t
    operator+(const T &off) const
    {
        _ptr_t ret(*this);
        ret += off;
        return ret;
    }

    inline typename Ptr::backend_type
    get_device_addr() const
    {
        ASSERTION(is_device_ptr());
        return ptrDev_.get();
    }

    inline hostptr_t
    get_host_addr() const
    {
        ASSERTION(is_host_ptr());
        return ptrHost_;
    }

    inline size_t
    get_offset() const
    {
        ASSERTION(is_device_ptr());
        return ptrDev_.offset();
    }

    inline C *
    get_hal_context()
    {
        return ctx_;
    }

    inline const C *
    get_hal_context() const
    {
        return ctx_;
    }

    inline bool
    is_host_ptr() const
    {
        return ctx_ == NULL;
    }

    inline bool
    is_device_ptr() const
    {
        return ctx_ != NULL;
    }
};

}}

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
