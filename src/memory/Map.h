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

#ifndef __MEMORY_MAP_H_
#define __MEMORY_MAP_H_

#include <memory/PageTable.h>

#include <paraver.h>
#include <util/Lock.h>

#include <cassert>
#include <set>
#include <map>

namespace gmac { namespace memory {

class Region;
typedef std::map<const void *, Region *> RegionMap;

class Map : public RegionMap {
protected:
    util::RWLock local;
    static RegionMap *__global;
    static unsigned count;
    static gmac::util::RWLock global;

    Region *localFind(const void *addr);
    static Region *globalFind(const void *addr);

    void clean();

    PageTable __pageTable;

public:
    Map();
    virtual ~Map();

    static void init();

    void realloc();

    void lock();
    void unlock();

    void insert(Region *i);

    Region *remove(void *addr);

    PageTable &pageTable();
    const PageTable &pageTable() const;

    template<typename T>
    T *find(const void *addr);
};

}}

#include "Map.ipp"

#endif
