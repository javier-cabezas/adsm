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

#ifndef __MEMORY_ALLOCATOR_CACHE_H_
#define __MEMORY_ALLOCATOR_CACHE_H_

#include <gmac/gmac.h>

#include <util/Private.h>
#include <util/Logger.h>
#include <memory/Manager.h>

#include <list>
#include <map>

namespace gmac { namespace memory { namespace allocator {

typedef std::list<void *> ObjectList;

class Arena : public util::Logger {
protected:
    void *ptr;
    size_t size;

    ObjectList _objects;

public:
    Arena(size_t objSize);
    ~Arena();

    void *address() const;
    const ObjectList &objects() const;

    bool full() const;
    bool empty() const;

    void *get();
    void put(void *obj);
};


class Cache : public util::Logger {
protected:
    size_t objectSize;
    size_t arenaSize;

    typedef std::map<void *, Arena *> ArenaMap;
    ArenaMap arenas;

public:
    Cache(size_t size);
    virtual ~Cache();

    static Cache &get(long key, size_t size);
    static void cleanup();

    void *get();
    void put(void *obj);

};

}}}

#include "Cache.ipp"

#endif
