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

#ifndef __MEMORY_MEMMANAGER_H_
#define __MEMORY_MEMMANAGER_H_

#include <util/Logger.h>

#include <memory/PageTable.h>

#include <stdint.h>

#include <iostream>
#include <iterator>
#include <map>


namespace gmac { namespace memory {


//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class Manager : public util::Logger {
protected:

#ifdef USE_MMAP
#ifdef ARCH_32BIT
    static const size_t mmSize = 0;
#else
	static const size_t mmSize = 0x10000000000;
#endif
#endif

    static int __count;
    static Manager *__manager;

	const PageTable &pageTable() const;

    Manager();
    ~Manager();
public:
    // Manager management
    static inline Manager *create();
    static inline void destroy();
    static inline Manager *get();

    // Memory management functions
    gmacError_t alloc(void **addr, size_t size);
#ifndef USE_MMAP
    gmacError_t globalAlloc(void **addr, size_t size);
#endif
    gmacErrot_t free (void *addr);

    // Coherence protocol interface
    gmacError_t adquire();
    gmacError_t release();
    void read(void *addr);
    void write(void *addr);
} };


#include "Manager.ipp"

#endif
