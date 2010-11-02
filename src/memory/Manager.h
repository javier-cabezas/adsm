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

#ifndef GMAC_MEMORY_MANAGER_H_
#define GMAC_MEMORY_MANAGER_H_

#include "config/common.h"
#include "include/gmac-types.h"
#include "util/Logger.h"
#include "util/Singleton.h"

namespace gmac {
class IOBuffer;
class Mode;

namespace memory {
class Protocol;

namespace __impl {
//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class GMAC_LOCAL Manager :
    public util::Logger,
    public util::Singleton<memory::Manager> {
	// Needed to let Singleton call the protected constructor
	friend class util::Singleton<memory::Manager>;
private:
#ifdef USE_VM
    void checkBitmapToHost();
    void checkBitmapToDevice();
#endif
protected:
    // TODO should we allow per-object protocol?
    memory::Protocol *protocol_;

    Manager();
    ~Manager();
public:
    //////////////////////////////
    // Memory management functions
    //////////////////////////////
    gmacError_t map(void *addr, size_t size, GmacProtection prot);
    gmacError_t unmap(void *addr, size_t size);
    gmacError_t alloc(void **addr, size_t size);
#ifndef USE_MMAP
    gmacError_t globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint);
    bool requireUpdate(gmac::memory::Block &block);
#endif
    gmacError_t free(void *addr);

    ///////////////////////////////
    // Coherence protocol interface
    ///////////////////////////////
    gmacError_t acquire();
    gmacError_t release();
    gmacError_t invalidate();

    bool read(void *addr);
    bool write(void *addr);

    /////////////////////////
    // Memory bulk operations
    /////////////////////////
    gmacError_t toIOBuffer(IOBuffer &buffer, const void *addr, size_t size);
    gmacError_t fromIOBuffer(void *addr, IOBuffer &buffer, size_t size);

    gmacError_t memcpy(void * dst, const void * src, size_t n);
    gmacError_t memset(void * dst, int c, size_t n);

    ///////////////////
    // Object migration
    ///////////////////
    gmacError_t moveTo(void * addr, Mode &mode);

    //////////////////
    // Mode management
    //////////////////
    gmacError_t removeMode(Mode &mode);

    ///////////////////////////////////////////////////
    // Direct access to protocol for internal functions
    ///////////////////////////////////////////////////
    gmac::memory::Protocol &protocol() const;
};

}}}

#include "Manager.ipp"

#include "memory/dbc/Manager.h"

#endif
