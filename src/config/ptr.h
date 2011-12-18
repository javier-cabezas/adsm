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

#ifndef GMAC_CONFIG_PTR_H_
#define GMAC_CONFIG_PTR_H_

template <typename BackendPtr>
class _common_ptr_t {
protected:
    BackendPtr devPtr_;
    unsigned asId_;

public:
    inline _common_ptr_t(BackendPtr ptr, unsigned asId = 0) :
        devPtr_(ptr),
        asId_(asId)
    {
    }

    inline _common_ptr_t(typename BackendPtr::base_type value, unsigned asId = 0) :
        devPtr_(value),
        asId_(asId)
    {
    }

    inline _common_ptr_t(const _common_ptr_t<BackendPtr> &ptr) :
        devPtr_(ptr.devPtr_),
        asId_(ptr.asId_)
    {
    }

    inline ~_common_ptr_t()
    {
    }

    inline _common_ptr_t &operator=(const _common_ptr_t<BackendPtr> &ptr)
    {
        if (this != &ptr) {
            devPtr_ = ptr.devPtr_;
            asId_  = ptr.asId_;
        }
        return *this;
    }

    inline bool operator==(const _common_ptr_t<BackendPtr> &ptr) const
    {
        return devPtr_ == ptr.devPtr_ && asId_ == ptr.asId_;
    }

    inline bool operator==(long i) const
    {
        return devPtr_ == i;
    }

    inline bool operator!=(const _common_ptr_t<BackendPtr> &ptr) const
    {
        return devPtr_ != ptr.devPtr_ || asId_ != ptr.asId_;
    }

    inline bool operator!=(long i) const
    {
        return devPtr_ != i;
    }

    inline bool operator<(const _common_ptr_t<BackendPtr> &ptr) const
    {
        return asId_ < ptr.asId_ || (asId_ == ptr.asId_ && devPtr_ < ptr.devPtr_);
    }

    template <typename T>
    inline _common_ptr_t &operator+=(const T &off)
    {
        devPtr_ += off;
        return *this;
    }

    template <typename T>
    inline const _common_ptr_t operator+(const T &off) const
    {
        _common_ptr_t ret(*this);
        ret += off;
        return ret;
    }

    inline typename BackendPtr::base_type get() const
    {
        return devPtr_.get();
    }

    inline size_t offset() const
    {
        return devPtr_.offset();
    }

    inline unsigned getPAddressSpace() const
    {
        return asId_;
    }
};


#endif /* PTR_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
