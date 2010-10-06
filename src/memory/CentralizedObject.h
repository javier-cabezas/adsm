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

#ifndef GMAC_MEMORY_CENTRALIZEDOBJECT_H_
#define GMAC_MEMORY_CENTRALIZEDOBJECT_H_

#include "core/Mode.h"
#include "memory/Block.h"
#include "memory/Object.h"

namespace gmac { namespace memory {
#ifndef USE_MMAP
class CentralizedObject : public Object {
public:
    CentralizedObject(size_t size);
    virtual ~CentralizedObject();

    virtual void init();
    virtual void fini();

    virtual void *device(void *addr) const;
    inline virtual Mode &owner() const { return gmac::Mode::current(); }

    inline gmacError_t toHost(Block &block, void *hostAddr = 0) const {
        Fatal("Trying to acquire a centralized object");
        return gmacErrorInvalidValue;
    }
    inline gmacError_t toDevice(Block &block) const {
        Fatal("Trying to release a centralized object");
        return gmacErrorInvalidValue;
    }

    inline bool local() const { return false; }
};
#endif

}}

#endif
