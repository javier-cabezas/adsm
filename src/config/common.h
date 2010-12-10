/* Copyright (c) 2009 University of Illinois
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


#ifndef GMAC_CONFIG_COMMON_H_
#define GMAC_CONFIG_COMMON_H_

#include "config/config.h"
#include "dbc/types.h"

#if defined(__GNUC__)
#include <stdint.h>
#elif defined(_MSC_VER)
typedef unsigned __int8 uint8_t;
typedef signed __int8 int8_t;
typedef unsigned __int16 uint16_t;
typedef signed __int16 int16_t;
typedef unsigned __int32 uint32_t;
typedef signed __int32 int32_t;
typedef unsigned __int64 uint64_t;
typedef signed __int64 int64_t;
#endif

#ifndef _MSC_VER
#define UNREFERENCED_PARAMETER(a)
#endif

typedef uint8_t *hostptr_t;

#ifdef USE_CUDA
#include <cuda.h>
struct accptr_t {
    CUdeviceptr ptr_;
    accptr_t() :
        ptr_(NULL) {}

    accptr_t(void * ptr) :
        ptr_(CUdeviceptr(ptr)) {}

    accptr_t(CUdeviceptr ptr) :
        ptr_(ptr) {}

    accptr_t(long int ptr) :
        ptr_(ptr) {}

    operator uint32_t()
    {
        return uint32_t(ptr_);
    }

    operator uint64_t()
    {
        return uint64_t(ptr_);
    }

    operator CUdeviceptr()
    {
        return ptr_;
    }

    operator const CUdeviceptr() const
    {
        return ptr_;
    }

    operator void *()
    {
        return (void *)(ptr_);
    }

    operator void *() const
    {
        return (void *)(ptr_);
    }
};

static inline
bool operator==(const accptr_t &ptr1, const void *ptr2)
{
    return (((const void *)ptr1.ptr_) == ptr2);
}

static inline
bool operator==(const accptr_t &ptr1, long int ptr2)
{
    return (((long int)ptr1.ptr_) != ptr2);
}

static inline
bool operator!=(const accptr_t &ptr1, const void *ptr2)
{
    return (((const void *)ptr1.ptr_) != ptr2);
}

static inline
bool operator!=(const accptr_t &ptr1, long int ptr2)
{
    return (((long int)ptr1.ptr_) != ptr2);
}

static inline
accptr_t operator>>(const accptr_t &ptr1, int shift)
{
    return accptr_t(ptr1.ptr_>> shift);
}

static inline
accptr_t operator>>(const accptr_t &ptr1, unsigned shift)
{
    return accptr_t(ptr1.ptr_>> shift);
}

static inline
accptr_t operator<<(const accptr_t &ptr1, int shift)
{
    return accptr_t(ptr1.ptr_<< shift);
}

static inline
accptr_t operator<<(const accptr_t &ptr1, unsigned shift)
{
    return accptr_t(ptr1.ptr_<< shift);
}

static inline
accptr_t operator+(const accptr_t &ptr1, int add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, long int add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, unsigned add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator+(const accptr_t &ptr1, size_t add)
{
    return accptr_t(ptr1.ptr_ + add);
}

static inline
accptr_t operator-(const accptr_t &ptr1, int sub)
{
    return accptr_t(ptr1.ptr_ - sub);
}

static inline
accptr_t operator-(const accptr_t &ptr1, long int add)
{
    return accptr_t(ptr1.ptr_ - add);
}

static inline
accptr_t operator-(const accptr_t &ptr1, unsigned sub)
{
    return accptr_t(ptr1.ptr_ - sub);
}

static inline
accptr_t operator-(const accptr_t &ptr1, size_t sub)
{
    return accptr_t(ptr1.ptr_ - sub);
}

#else
#ifdef USE_OPENCL
#include <opencl.h>
class _opencl_ptr_t {
    cl_mem base_;
    size_t offset_;
public:
    _opencl_ptr_t(cl_mem base, size_t offset) :
        base_(base),
        offset_(offset)
    {
    }

    _opencl_ptr_t(const _opencl_ptr_t &ptr) :
        base_(ptr.base_),
        offset_(ptr.offset_)
    {
    }

    const _opencl_ptr_t & operator=(const _opencl_ptr_t &ptr)
    {
        base_   = ptr.base_;
        offset_ = ptr.offset_;
        return *this;
    }

    const _opencl_ptr_t operator+(unsigned off)
    {
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ + off;
        return tmp;
    }

    const _opencl_ptr_t operator-(unsigned off)
    {
        ASSERTION(off < offset_);
        _opencl_ptr_t tmp;
        tmp.base_   = base_;
        tmp.offset_ = offset_ - off;
        return tmp;
    }
};

typedef _opencl_ptr_t accptr_t;
#else
#error "No programming model back-end specified"
#endif
#endif


namespace __impl {
#if defined(GMAC_DLL)
    void enterGmac();
    void enterGmacExclusive();
    void exitGmac();
    char inGmac();
#endif

    namespace cuda {}
    namespace core {}
    namespace util {}
    namespace memory {
        namespace protocol {}
    }
    namespace trace {}
}

#ifdef USE_DBC
namespace __dbc {
#if defined(GMAC_DLL)
    using __impl::enterGmac;
    using __impl::enterGmacExclusive;
    using __impl::exitGmac;
    using __impl::inGmac;
#endif

    namespace cuda {}
    namespace core {
        // Singleton classes need to be predeclared
        class Process;
    }
    namespace util {}
    namespace memory {
        // Singleton classes need to be predeclared
        class Manager;
        namespace protocol {}
    }
    namespace trace = __impl::trace;
}
#endif
#ifdef USE_DBC
namespace gmac = __dbc;

#define DBC_FORCE_TEST(c) virtual void __dbcForceTest(c &o) = 0;
#define DBC_TESTED(c)             void __dbcForceTest(c &) {}
#else
namespace gmac = __impl;

#define DBC_FORCE_TEST(c)
#define DBC_TESTED(c)
#endif

#include "include/gmac/visibility.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
