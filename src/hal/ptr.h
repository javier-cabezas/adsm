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
    _ptr_t();
    _ptr_t(Ptr ptr, C *ctx);
    _ptr_t(typename Ptr::backend_type value, C *ctx);
    explicit _ptr_t(hostptr_t ptr);

    _ptr_t(const _ptr_t &ptr);

    operator bool() const;

    _ptr_t &
    operator=(const _ptr_t &ptr);
    bool
    operator==(const _ptr_t &ptr) const;
    bool
    operator==(long i) const;
    bool
    operator!=(const _ptr_t &ptr) const;
    bool
    operator!=(long i) const;
    bool
    operator<(const _ptr_t &ptr) const;
    template <typename T>
    _ptr_t &
    operator+=(const T &off);
    template <typename T>
    const _ptr_t
    operator+(const T &off) const;

    typename Ptr::backend_type
    get_device_addr() const;
    hostptr_t
    get_host_addr() const;

    size_t
    get_offset() const;

    C *
    get_context();

    const C *
    get_context() const;

    bool
    is_host_ptr() const;

    bool
    is_device_ptr() const;
};

}}

#include "ptr-impl.h"

#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
