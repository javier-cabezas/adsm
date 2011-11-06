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

template <typename D, typename B, typename I>
class ptr_t {
protected:
    typename B::alloc devPtr_;
    typename I::context *ctx_;

public:
    inline ptr_t(typename B::alloc ptr, typename I::context *ctx = NULL) :
        devPtr_(ptr),
        ctx_(ctx)

    {
    }

    inline ptr_t(typename B::alloc value, typename I::context *ctx = NULL) :
        devPtr_(value),
        ctx_(ctx)
    {
    }

    inline ptr_t(const ptr_t<typename B::alloc> &ptr) :
        devPtr_(ptr.devPtr_),
        ctx_(ptr.ctx_)
    {
    }

    inline ~ptr_t()
    {
    }

    inline ptr_t &operator=(const ptr_t<typename B::alloc> &ptr)
    {
        if (this != &ptr) {
            devPtr_ = ptr.devPtr_;
            ctx_    = ptr.ctx_;
        }
        return *this;
    }

    inline bool operator==(const ptr_t<typename B::alloc> &ptr) const
    {
        return devPtr_ == ptr.devPtr_ && ctx_ == ptr.ctx_;
    }

    inline bool operator==(long i) const
    {
        return devPtr_ == i;
    }

    inline bool operator!=(const ptr_t<typename B::alloc> &ptr) const
    {
        return devPtr_ != ptr.devPtr_ || ctx_ != ptr.ctx_;
    }

    inline bool operator!=(long i) const
    {
        return devPtr_ != i;
    }

    inline bool operator<(const ptr_t<typename B::alloc> &ptr) const
    {
        return ctx_ < ptr.ctx_ || (ctx_ == ptr.ctx_ && devPtr_ < ptr.devPtr_);
    }

    template <typename T>
    inline ptr_t &operator+=(const T &off)
    {
        devPtr_ += off;
        return *this;
    }

    template <typename T>
    inline const ptr_t operator+(const T &off) const
    {
        ptr_t ret(*this);
        ret += off;
        return ret;
    }

    inline typename typename B::alloc get() const
    {
        return devPtr_.get();
    }

    inline size_t offset() const
    {
        return devPtr_.offset();
    }

    inline context_t &get_hal_context()
    {
        return ctx_;
    }

    inline const context_t &get_hal_context() const
    {
        return ctx_;
    }
};

}}

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
