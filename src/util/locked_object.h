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

#ifndef GMAC_UTIL_LOCKED_OBJECT_H_
#define GMAC_UTIL_LOCKED_OBJECT_H_

#include <utility>

#include "lock.h"

namespace __impl { namespace util {

template <typename T>
class locked_object_ref :
    public T::locker_type {

    typedef typename T::locker_type parent;

    T *obj_;

public:
    locked_object_ref(T &obj) :
        obj_(&obj)
    {
        parent::lock(*obj_);
    }

    locked_object_ref(locked_object_ref &&l) :
        obj_(l.obj_)
    {
        l.obj_ = NULL;
    }

    ~locked_object_ref()
    {
        if (obj_ != NULL) {
            parent::unlock(*obj_);
        }
    }

    locked_object_ref &
    operator=(locked_object_ref &&l)
    {
        if (&l != this) {
            if (obj_ != NULL) {
                parent::unlock(*obj_);
            }

            obj_ = l.obj_;
            l.obj_ = NULL;
        }

        return *this;
    }

    T &operator*()
    {
        ASSERTION(obj_ != NULL);
        return *obj_;
    }

    const T &operator*() const
    {
        ASSERTION(obj_ != NULL);
        return *obj_;
    }
};

template <typename T>
class locked_object_ptr :
    public T::element_type::locker_type {

    typedef typename T::element_type::locker_type parent;

    T obj_;
    bool unlockOnDestruction_;

public:
    locked_object_ptr(T obj) :
        obj_(obj),
        unlockOnDestruction_(true)
    {
        parent::lock(*obj_);
    }

    locked_object_ptr(locked_object_ptr &&l) :
        obj_(std::move(l.obj_))
    {
        l.unlockOnDestruction_ = false;
    }

    ~locked_object_ptr()
    {
        ASSERTION(!unlockOnDestruction_ || obj_);
        if (obj_) {
            parent::unlock(*obj_);
        }
    }

    locked_object_ptr &
    operator=(locked_object_ptr &&l)
    {
        if (&l != this) {
            if (obj_ != NULL) {
                parent::unlock(*obj_);
            }

            obj_ = l.obj_;
            l.obj_ = NULL;
        }

        return *this;
    }

    T operator*() const
    {
        return obj_;
    }
};

template <typename T>
class locked_object :
    public conditional<__impl::util::is_any_ptr<T>::value,
    		           locked_object_ptr<T>,
    		           locked_object_ref<T> >::type {

    typedef typename conditional<__impl::util::is_any_ptr<T>::value,
    		                     locked_object_ptr<T>,
    		                     locked_object_ref<T> >::type parent;

    locked_object(locked_object &l);

    locked_object &operator=(const locked_object &l);

public:
    locked_object(T &obj) :
        parent(obj)
    {
    }

    locked_object(locked_object &&l) :
        parent(std::move(l))
    {
    }
};

}}

#endif /* GMAC_UTIL_LOCKED_OBJECT_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
