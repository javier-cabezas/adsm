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


#ifndef GMAC_UTIL_REUSABLEOBJECT_H_
#define GMAC_UTIL_REUSABLEOBJECT_H_

#include <cstddef>

#include "config/common.h"

namespace __impl { namespace util {

/*! \todo Delete ? */
template <typename Type>
class GMAC_LOCAL Pool {
    template <typename Type2> friend class ReusableObject;
public:
    Pool()
        : freeList(NULL)
        {}

    ~Pool()
        {
            Object * next, * tmp;
            for (next = freeList; next; next = tmp) {
                tmp = next->next;
                delete next;
            }
        }

    union Object {
        char shit[sizeof(Type)];
        Object * next;
    };

private:
    Object * freeList;

};

/*! \todo Delete ? */
template <typename Type>
class GMAC_LOCAL ReusableObject {
public:
    void * operator new (size_t bytes)
        {
            typename Pool<Type>::Object * res = pool.freeList;
            return res? (pool.freeList = pool.freeList->next, res) : new typename Pool<Type>::Object;
        }

    void operator delete (void *ptr)
        {
            ((typename Pool<Type>::Object *) ptr)->next = pool.freeList;
            pool.freeList = (typename Pool<Type>::Object *) ptr;
        }

protected:
    static Pool<Type> pool;
};

// Static initialization

template<class T> GMAC_LOCAL Pool<T> 
ReusableObject<T>::pool;

}}

#endif /* CYCLE_UTILS_REUSABLE_OBJECT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=100 foldmethod=marker expandtab cindent cinoptions=p5,t0,(0: */
