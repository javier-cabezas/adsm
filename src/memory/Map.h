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

#include "Bitmap.h"

#include <util/Lock.h>
#include <util/Logger.h>

#include <set>
#include <map>

namespace gmac { namespace memory {

class Object;
class ObjectMap : protected util::RWLock, public std::map<const void *, Object *> {
protected:
    friend class Map;
public:
    ObjectMap(const char *name) : util::RWLock(name) {};
};

class Map : public ObjectMap, public util::Logger {
protected:
    Object *mapFind(ObjectMap &map, const void *addr);
    inline Object *localFind(const void *addr) {
        return mapFind(*this, addr);
    }
    Object *globalFind(const void *addr);
#ifndef USE_MMAP
    Object *sharedFind(const void *addr);
#endif

    void clean();

public:
    Map(const char *name) : ObjectMap(name) {};
    virtual ~Map();

    void insert(Object *obj);
    void remove(Object *obj);
#ifndef USE_MMAP
    void insertShared(Object *obj);
    Object *removeShared(const void *addr);

    void insertGlobal(Object *obj);
    void removeGlobal(Object *obj);
#endif

    Object *find(const void *addr);
    
};

}}

#endif
