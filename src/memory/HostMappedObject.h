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

#ifndef GMAC_MEMORY_HOSTMAPPEDOBJECT_H_
#define GMAC_MEMORY_HOSTMAPPEDOBJECT_H_

#include "util/Lock.h"
#include "util/Reference.h"
#include <map>

namespace __impl { 

namespace core {
	class Mode;
}

namespace memory {

void *HostMappedAlloc(size_t size);
void HostMappedFree(void *addr);
void *HostMappedPtr(const void *addr);

class HostMappedObject;

class GMAC_LOCAL HostMappedSet : protected std::map<void *, HostMappedObject *>, 
    public gmac::util::RWLock {
protected:
    friend class HostMappedObject;

    typedef std::map<void *, HostMappedObject *> Parent;
    bool insert(HostMappedObject *object);
    HostMappedObject *get(void *addr) const;
public:
    HostMappedSet();
    ~HostMappedSet();
    bool remove(void *addr);
};

class GMAC_LOCAL HostMappedObject : public util::Reference {
protected:
    uint8_t *addr_;
    size_t size_;

    static HostMappedSet set_;    
public:
	HostMappedObject(size_t size);
    virtual ~HostMappedObject();
    
    void *addr() const;
    size_t size() const;
    void *deviceAddr(const void *addr) const;

    static void remove(void *addr);
    static HostMappedObject *get(const void *addr);
};

}}

#include "HostMappedObject-impl.h"


#endif
