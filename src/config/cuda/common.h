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

#ifndef GMAC_CONFIG_CUDA_COMMON_H_
#define GMAC_CONFIG_CUDA_COMMON_H_

#include <cuda.h>

typedef CUcontext AddressSpace;

struct accptr_t {
    CUdeviceptr ptr_;
    unsigned pasId_;

    template<typename T>
    inline accptr_t(T * ptr) :
        ptr_(CUdeviceptr(ptr)),
        pasId_(0)
    {}

    inline accptr_t(CUdeviceptr ptr) :
        ptr_(ptr),
        pasId_(0)
    {}
    
    inline operator CUdeviceptr() const { return ptr_; }

    inline operator void *() const { return (void *)(ptr_); }

    inline void *get() const { return (void *)(ptr_); }

    inline
    bool operator==(accptr_t ptr)
    {
        return this->ptr_ == ptr.ptr_ && this->pasId_ == ptr.pasId_;
    }

    inline
    bool operator!=(accptr_t ptr)
    {
        return this->ptr_ != ptr.ptr_ || this->pasId_ != ptr.pasId_;
    }

    template <typename T>
    inline
    bool operator==(T ptr)
    {
        return (((T)this->ptr_) == ptr);
    }

    template <typename T>
    inline
    bool operator!=(T ptr)
    {
        return (((T)this->ptr_) != ptr);
    }
};

inline
static bool operator<(const accptr_t &a, const accptr_t &b)
{
    return a.pasId_ < b.pasId_ || (a.pasId_ == b.pasId_ && a.ptr_ < b.ptr_);
}



template <typename T>
static inline
accptr_t operator+(const accptr_t &a, T b)
{
    accptr_t ret(a.ptr_ + CUdeviceptr(b));
    ret.pasId_ = a.pasId_;
    return ret;
}

template <typename T>
static inline
accptr_t operator-(const accptr_t &a, T b)
{
    accptr_t ret(a.ptr_ - CUdeviceptr(b));
    ret.pasId_ = a.pasId_;
    return ret;
}

#endif
