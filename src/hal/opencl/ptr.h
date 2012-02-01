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

#ifndef GMAC_HAL_OPENCL_H_
#define GMAC_HAL_OPENCL_H_

#if defined(__APPLE__)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include <cassert>
#include <cstdlib>

#include "hal/ptr.h"

namespace __impl { namespace hal {

class _opencl_ptr_t {
private:
    cl_mem ptr_;
    off_t off_;

public:
    typedef cl_mem backend_type;

    inline explicit _opencl_ptr_t(cl_mem mem) :
        ptr_(mem),
        off_(0)
    {
    }

    inline _opencl_ptr_t(const _opencl_ptr_t &ptr) :
        ptr_(ptr.ptr_),
        off_(ptr.off_)
    {
    }

    inline _opencl_ptr_t &operator=(const _opencl_ptr_t &ptr)
    {
        if (this != &ptr) {
            ptr_ = ptr.ptr_;
            off_ = ptr.off_;
        }
        return *this;
    }

    inline bool operator==(const _opencl_ptr_t &ptr) const
    {
        return ptr_ == ptr.ptr_ && off_ == ptr.off_;
    }

    inline bool operator==(long i) const
    {
        return ptr_ == cl_mem(i);
    }

    inline bool operator!=(const _opencl_ptr_t &ptr) const
    {
        return ptr_ != ptr.ptr_ || off_ != ptr.off_;
    }

    inline bool operator!=(long i) const
    {
        return ptr_ != cl_mem(i);
    }

    inline bool operator<(const _opencl_ptr_t &ptr) const
    {
        return ptr_ < ptr.ptr_ || (ptr_ == ptr.ptr_ && off_ < ptr.off_);
    }

    template <typename T>
    inline _opencl_ptr_t &operator+=(const T &off)
    {
        off_ += off;
        return *this;
    }

    template <typename T>
    inline _opencl_ptr_t &operator-=(const T &off)
    {
        ASSERTION(off_ >= off, "Generating a pointer before the base allocation");
        off_ -= off;
        return *this;
    }

    inline cl_mem get() const
    {
        return ptr_;
    }

    inline size_t offset() const
    {
        return off_;
    }
};

}}

#endif
