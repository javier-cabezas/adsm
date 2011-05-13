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

#ifndef GMAC_CONFIG_OPENCL_COMMON_H_
#define GMAC_CONFIG_OPENCL_COMMON_H_

#include <CL/cl.h>

#include <cassert>
#include <cstdlib>

typedef cl_context AddressSpace;

struct _opencl_ptr_t {
    cl_mem base_;
    size_t offset_;
    unsigned pasId_;

    inline _opencl_ptr_t() :
        base_(0),
        offset_(0),
        pasId_(0)
    { }

    inline _opencl_ptr_t(cl_mem base) :
        base_(base),
        offset_(0),
        pasId_(0)
    {}

    inline _opencl_ptr_t(cl_mem base, size_t offset) :
        base_(base),
        offset_(offset),
        pasId_(0)
    { }
    inline _opencl_ptr_t(const _opencl_ptr_t &ptr) :
        base_(ptr.base_),
        offset_(ptr.offset_),
        pasId_(0)
    { }

    inline const _opencl_ptr_t & operator=(const _opencl_ptr_t &ptr) {
        base_   = ptr.base_;
        offset_ = ptr.offset_;
        return *this;
    }

    template <typename T>
    inline const _opencl_ptr_t operator+(const T off) const {
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ + off;
        return tmp;
    }

    template <typename T>
    inline const _opencl_ptr_t operator-(const T off) const {
        assert(off < offset_);
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ - off;
        return tmp;
    }

    inline const bool operator!=(const hostptr_t addr) const {
        return base_ != cl_mem(addr - offset_);
    }

    inline const bool operator!=(const _opencl_ptr_t ptr) const {
        return base_ != ptr.base_ || offset_ != ptr.offset_;
    }

    inline operator void*() { return (void *)base_; }

    inline cl_mem get() const { return base_; }
};

typedef _opencl_ptr_t accptr_t;

inline
static bool operator<(const accptr_t &ptr, const accptr_t &ptr2)
{
    return ptr.base_ < ptr2.base_;
}

extern const _opencl_ptr_t nullaccptr;

#endif
