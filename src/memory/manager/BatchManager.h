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

#ifndef __MEMORY_BATCHMANAGER_H_
#define __MEMORY_BATCHMANAGER_H_

#include "Manager.h"
#include "Region.h"

#include <stdint.h>

namespace gmac { namespace memory { namespace manager {
//! Batch Memory Manager

//! The Batch Memory Manager moves all data just before and
//! after a kernel call
class BatchManager : public Manager {
public:
	BatchManager() : Manager() { }

#if 0
	gmacError_t malloc(void ** addr, size_t count);
	gmacError_t globalMalloc(void ** addr, size_t count);
    void free(void * addr);
#endif
    void invalidate();
    void invalidate(const RegionSet & regions);
    void flush();
    void flush(const RegionSet & regions);

	void invalidate(const void *, size_t);
	void flush(const void *, size_t);

	void map(Context *, Region *, void *);

    bool touch(Region * r) { return true; }
};

}}}

#include "BatchManager.ipp"

#endif
