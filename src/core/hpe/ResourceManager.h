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

#ifndef GMAC_CORE_HPE_RESOURCE_MANAGER_H_
#define GMAC_CORE_HPE_RESOURCE_MANAGER_H_

#include <vector>

#include "config/common.h"
#include "config/order.h"
#include "core/ResourceManager.h"
#include "util/loader.h"

namespace __impl {

namespace core {

namespace hpe {

class Process;

/** Represents the resources used by a running process */
class GMAC_LOCAL ResourceManager :
    public core::ResourceManager {
    // Needed to let Singleton call the protected constructor
    friend class Process;
protected:
    /**
     * Constructs the resource manager
     */
    ResourceManager();
    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~ResourceManager();

public:
    gmacError_t registerAccelerator(Accelerator &acc);

    GmacAddressSpace newAddressSpace(unsigned accId);

    GmacVirtualDevice newVirtualDevice(GmacAddressSpace aSpace);

    gmacError_t hostAlloc(core::Mode &mode, hostptr_t &addr, size_t size);

    gmacError_t hostFree(core::Mode &mode, hostptr_t addr);

    accptr_t hostMapAddr(core::Mode &mode, const hostptr_t addr);

    gmacError_t map(core::Mode &mode, accptr_t &dst, hostptr_t src, size_t count, unsigned align = 1);

    gmacError_t unmap(core::Mode &mode, hostptr_t addr, size_t count);

    gmacError_t copyToAccelerator(core::Mode &mode, accptr_t acc, const hostptr_t host, size_t count);

    gmacError_t copyToHost(core::Mode &mode, hostptr_t host, const accptr_t acc, size_t count);

    gmacError_t copyAccelerator(core::Mode &mode, accptr_t dst, const accptr_t src, size_t count);

    core::IOBuffer &createIOBuffer(core::Mode &mode, size_t count, GmacProtection prot);

    void destroyIOBuffer(core::Mode &mode, core::IOBuffer &buffer);

    gmacError_t bufferToAccelerator(core::Mode &mode, accptr_t dst, core::IOBuffer &buffer, size_t count, size_t off = 0);

    gmacError_t acceleratorToBuffer(core::Mode &mode, core::IOBuffer &buffer, const accptr_t dst, size_t count, size_t off = 0);

    gmacError_t memset(core::Mode &mode, accptr_t addr, int c, size_t size);
};

}}}

#endif
