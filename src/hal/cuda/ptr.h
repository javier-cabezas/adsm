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

#ifndef GMAC_HAL_CUDA_H_
#define GMAC_HAL_CUDA_H_

#include <cstdio>
#include <cuda.h>

#include "hal/ptr.h"

namespace __impl { namespace hal {

class _cuda_ptr_t {
private:
    CUdeviceptr base_;
    size_t off_;

public:
    typedef CUdeviceptr backend_type;

    inline explicit
    _cuda_ptr_t(CUdeviceptr ptr) :
        base_(ptr),
        off_(0)
    {
    }

#if 0
    inline operator CUdeviceptr() const
    {
        return base_;
    }
#endif

    inline _cuda_ptr_t &
    operator=(const _cuda_ptr_t &ptr)
    {
        if (this != &ptr) {
            base_ = ptr.base_;
            off_  = ptr.off_;
        }
        return *this;
    }

    inline bool
    operator==(const _cuda_ptr_t &ptr) const
    {
        return this->base_ == ptr.base_ &&
               this->off_  == ptr.off_;
    }

    inline bool
    operator!=(const _cuda_ptr_t &ptr) const
    {
        return this->base_ != ptr.base_ ||
               this->off_  != ptr.off_;
    }

    inline bool
    operator<(const _cuda_ptr_t &ptr) const
    {
        return base_ < ptr.base_ || (base_ == ptr.base_ && off_ < ptr.off_);
    }

    inline bool
    operator<=(const _cuda_ptr_t &ptr) const
    {
        return base_ < ptr.base_ || (base_ == ptr.base_ && off_ <= ptr.off_);
    }

    inline bool
    operator>(const _cuda_ptr_t &ptr) const
    {
        return base_ > ptr.base_ || (base_ == ptr.base_ && off_ > ptr.off_);
    }

    inline bool
    operator>=(const _cuda_ptr_t &ptr) const
    {
        return base_ > ptr.base_ || (base_ == ptr.base_ && off_ >= ptr.off_);
    }

    template <typename T>
    inline _cuda_ptr_t &
    operator+=(const T &off)
    {
        off_ += off;
        return *this;
    }

    template <typename T>
    inline _cuda_ptr_t &
    operator-=(const T &off)
    {
        ASSERTION(off_ >= off);
        off_ -= off;
        return *this;
    }

    inline CUdeviceptr
    get_backend() const
    {
        return base_;
    }

    inline void *
    get() const
    {
        return (void *)(base_ + off_);
    }

    inline size_t
    offset() const
    {
        return off_;
    }
};

}}

#endif
