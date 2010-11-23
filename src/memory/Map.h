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

#ifndef GMAC_MEMORY_MAP_H_
#define GMAC_MEMORY_MAP_H_

#include <map>
#include <set>

#include "config/common.h"
#include "util/Lock.h"


namespace __impl {

namespace core {
class Mode;
class Process;
}

namespace memory {
class Object;
class Protocol;

class GMAC_LOCAL ObjectMap : 
	protected gmac::util::RWLock, protected std::map<const void *, Object *> {
public:
    typedef gmacError_t(Object::*ObjectOp)(void) const;
    typedef void (Object::*ModeOp)(const core::Mode &);
protected:
	friend class Map;
    typedef std::map<const void *, Object *> Parent;

    const Object *mapFind(const void *addr, size_t size) const;
public:
    ObjectMap(const char *name);
    virtual ~ObjectMap();
    size_t size() const;

	virtual bool insert(Object &obj);
	virtual bool remove(const Object &obj);
	virtual const Object *get(const void *addr, size_t size) const;

    size_t memorySize() const;
    void forEach(ObjectOp op) const;
    void forEach(const core::Mode &mode, ModeOp op) const;

    //void reallocObjects(gmac::core::Mode &mode);
};
 
class GMAC_LOCAL Map : public memory::ObjectMap {
protected:
    core::Mode &parent_;

    const Object *get(const ObjectMap &map, const uint8_t *&base, 
        const void *addr, size_t size) const;
public:
    Map(const char *name, core::Mode &parent);
    virtual ~Map();
    
    // Do not allow to copy memory maps
	Map &operator =(const Map &);

    bool insert(Object &obj);
    bool remove(const Object &obj);
	virtual const Object *get(const void *addr, size_t size) const;

    static void insertOrphan(Object &obj);

	static void addOwner(core::Process &proc, core::Mode &mode);
	static void removeOwner(core::Process &proc, const core::Mode &mode);

    //void makeOrphans();
};

}}

#endif
