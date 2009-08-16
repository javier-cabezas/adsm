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

#ifndef __MEMORY_PROTREGION_H_
#define __MEMORY_PROTREGION_H_

#include "MemRegion.h"
#include "MemHandler.h"

#include <os/Memory.h>

#include <signal.h>

namespace gmac {
//! Protected Memory Region
class ProtRegion : public MemRegion {
protected:
	bool dirty;
	bool present;

	static unsigned count;
	static struct sigaction defaultAction;
	static void setHandler(void);
	static void restoreHandler(void);
	static void segvHandler(int, siginfo_t *, void *);
public:
	ProtRegion(void *addr, size_t size);
	virtual ~ProtRegion();

	inline virtual void read(void *addr) {
		MemHandler::get()->read(this, addr);
	}
	inline virtual void write(void *addr) {
		MemHandler::get()->write(this, addr);
	}

	inline virtual void invalidate(void) {
		present = dirty = false;
		assert(Memory::protect(__void(addr), size, PROT_NONE) == 0);
	}
	inline virtual void readOnly(void) {
		present = true;
		dirty = false;
		assert(Memory::protect(__void(addr), size, PROT_READ) == 0);
	}
	inline virtual void readWrite(void) {
		present = dirty = true;
		assert(Memory::protect(__void(addr), size, PROT_READ | PROT_WRITE) == 0);
	}

	virtual bool isDirty() const { return dirty; }
	virtual bool isPresent() const { return present; }
};
};

#endif
