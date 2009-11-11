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

#ifndef __MEMORY_PAGETABLE_H_
#define __MEMORY_PAGETABLE_H_

#include <config.h>
#include <threads.h>
#include <debug.h>
#include <paraver.h>

#include <memory/VM.h>

#include <stdint.h>
#include <math.h>

#include <cassert>


namespace gmac { namespace memory {

//! Page Table 

//! Software Virtual Memory Table to keep translation from
// CPU to accelerator memory addresses
class PageTable {
protected:

private:
	static const char *pageSizeVar;
	static const size_t defaultPageSize = 2 * 1024 * 1024;

	static const unsigned long dirShift = 30;
	static const unsigned long rootShift = 39;

	MUTEX(mutex);
	inline void lock() { 
		enterLock(pageTable);
		MUTEX_LOCK(mutex);
		exitLock();
	}
	inline void unlock() { MUTEX_UNLOCK(mutex); }

	static size_t pageSize;
	static size_t tableShift;

	bool _clean;
	bool _valid;

	size_t pages;

	typedef vm::Table<vm::addr_t> Table;
	typedef vm::Table<Table> Directory;
	vm::Table<Directory> rootTable;

	inline int entry(const void *addr, unsigned long shift,
			size_t size) const {
		unsigned long n = (unsigned long)(addr);
		return (n >> shift) & (size - 1);
	}

	inline int offset(const void *addr) const {
		unsigned long n = (unsigned long)(addr);
		return n & (pageSize - 1);
	}

	void update();

	void deleteDirectory(Directory *dir);

	void flushDirectory(Directory &dir);
	void syncDirectory(Directory &dir);
	void sync();
	
public:
	PageTable();
	virtual ~PageTable();

	inline void realloc() { rootTable.realloc(); }

	void insert(void *host, void *dev);
	void remove(void *host);
	const void *translate(const void *host) {
		return translate((void *)host);
	}
	void *translate(void *host);

	size_t getPageSize() const { return pageSize; }
	size_t getTableShift() const { return tableShift; }
	size_t getTableSize() const { return (1 << dirShift) / pageSize; }

	void *flush();
	void invalidate() { _valid = false; }
	bool dirty(void *);
	void clear(void *);
};

}};
#endif
