/* Copyright (c) 2009, 2010, 2011 University of Illinois
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

#ifndef GMAC_CORE_RESOURCE_MANAGER_H_
#define GMAC_CORE_RESOURCE_MANAGER_H_

#include "config/common.h"
#include "config/order.h"
#include "util//loader.h"
#include "util/Singleton.h"

#include <vector>

namespace __impl {

namespace core {

/** Represents the resources used by a running process */
class GMAC_LOCAL ResourceManager  {
    // Needed to let Singleton call the protected constructor
protected:
    /**
     * Constructs the process
     */
    ResourceManager();
    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~ResourceManager();

public:
    virtual gmacError_t hostAlloc(Mode &mode, hostptr_t &addr, size_t size) = 0;

    virtual gmacError_t hostFree(Mode &mode, hostptr_t addr) = 0;

    virtual accptr_t hostMapAddr(Mode &mode, const hostptr_t addr) = 0;

    virtual gmacError_t map(Mode &mode, accptr_t &dst, hostptr_t src, size_t count, unsigned align = 1) = 0;

    virtual gmacError_t unmap(Mode &mode, hostptr_t addr, size_t count) = 0;

    virtual gmacError_t copyToAccelerator(Mode &mode, accptr_t acc, const hostptr_t host, size_t count) = 0;

    virtual gmacError_t copyToHost(Mode &mode, hostptr_t host, const accptr_t acc, size_t count) = 0;

    virtual gmacError_t copyAccelerator(Mode &mode, accptr_t dst, const accptr_t src, size_t count) = 0;

    virtual IOBuffer &createIOBuffer(Mode &mode, size_t count, GmacProtection prot) = 0;

    virtual void destroyIOBuffer(Mode &mode, IOBuffer &buffer) = 0;

    virtual gmacError_t bufferToAccelerator(Mode &mode, accptr_t dst, IOBuffer &buffer, size_t count, size_t off = 0) = 0;

    virtual gmacError_t acceleratorToBuffer(Mode &mode, IOBuffer &buffer, const accptr_t dst, size_t count, size_t off = 0) = 0;

    virtual gmacError_t memset(Mode &mode, accptr_t addr, int c, size_t size) = 0;
};

}}

#endif
