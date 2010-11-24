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


//! A map of objects that is not bound to any Mode
class GMAC_LOCAL ObjectMap : 
	protected gmac::util::RWLock, protected std::map<const void *, Object *> {
public:
    typedef gmacError_t(Object::*ObjectOp)(void) const;
    typedef void (Object::*ModeOp)(const core::Mode &);
protected:
	friend class Map;
    typedef std::map<const void *, Object *> Parent;

    //! Find an object in the map
    /*!
        \param addr Starting memory address within the object to be found
        \size Size (in bytes) of the memory range where the object can be found
        \return First object inside the memory range. NULL if no object is found
    */
    const Object *mapFind(const void *addr, size_t size) const;
public:
    //! Default constructor
    /*!
        \param name Name of the object map used for tracing
    */
    ObjectMap(const char *name);

    //! Default destructor
    virtual ~ObjectMap();

    //! Get the number of objects in the map
    /*!
        \return Number of objects in the map
    */
    size_t size() const;

    //! Insert an object in the map
    /*!
        \param obj Object to insert in the map
        \return True if the object was successfuly inserted
    */
	virtual bool insert(Object &obj);

    //! Remove an object from the map
    /*!
        \param obj Object to remove from the map
        \return True if the object was successfuly removed
    */
	virtual bool remove(const Object &obj);

    //! Find the firs object in a memory range
    /*!
        \param addr Starting address of the memory range where the object is located
        \param size Size (in bytes) of the memory range where the object is located
        \raturn First object within the memory range. NULL if no object is found
    */
	virtual const Object *get(const void *addr, size_t size) const;

    //! Get the amount of memory consumed by all objects in the map
    /*!
        \return Size (in bytes) of the memory consumed by all objects in the map
    */
    size_t memorySize() const;

    //! Invoke a memory operation over all the objects in the map
    /*!
        \param op Memory operation to be executed
        \sa __impl::memory::Object::acquire
        \sa __impl::memory::Object::toHost
        \sa __impl::memory::Object::toDevice
    */
    void forEach(ObjectOp op) const;

    //! Execute a mode operation over all the objects in the map
    /*!
        \param op Mode operation to be executed        
        \sa __impl::memory::Object::removeOwner
    */
    void forEach(const core::Mode &mode, ModeOp op) const;

    //void reallocObjects(gmac::core::Mode &mode);
};
 
//! An object map associated to an execution mode
class GMAC_LOCAL Map : public memory::ObjectMap {
protected:
    //! Execution mode owning this map
    core::Mode &parent_;

    //! Get the first object in an object map that is in a memory range whose starting memory address is bellow a base
    /*!
        \param map Object map to look for objects
        \param base Base memory address to filter out objects
        \param addr Starting address of the memory range to look for objects
        \param size Size (in bytes) of the memory range to look for objects
        \return First object in the memory range whose starting address is bellow the base
    */
    const Object *get(const ObjectMap &map, const uint8_t *&base, 
        const void *addr, size_t size) const;
public:
    //! Default constructor
    /*!
        \param name Name of the object map used for tracing
        \param parent Mode that owns the map
    */
    Map(const char *name, core::Mode &parent);

    //! Default destructor
    virtual ~Map();
    
    //! Null assigment operator to prevent assigment of memory maps
	Map &operator =(const Map &);

    //! Insert an object in the map and the global process map where all objects are registered
    /*!
        \param obj Object to remove from the map
        \return True if the object was successfuly removed
    */
    bool insert(Object &obj);

    //! Remove an object from the map and from the global process map where all objects are registered
    /*!
        \param obj Object to remove from the map
        \return True if the object was successfuly removed
    */
    bool remove(const Object &obj);

    //! Find the first object in a memory range in this map or on the global and shared process object maps
    /*!
        \param addr Starting address of the memory range where the object is located
        \param size Size (in bytes) of the memory range where the object is located
        \raturn First object within the memory range. NULL if no object is found
    */
	virtual const Object *get(const void *addr, size_t size) const;

    //! Remove object from this map and from the global process map and add that object to the process orphan object map
    /*!
        \param obj Memory object to be removed from the current and from the global process map and inserted in the orphan object map
    */
    static void insertOrphan(Object &obj);

    //! Add an owner to all global process objects
    /*!
        \param proc Process whose global objects will be the owner added to
        \param mode Owner to be added to global objects
    */
	static void addOwner(core::Process &proc, core::Mode &mode);

    //! Remove an owner to all global process objects
    /*!
        \param proc Process whose global objects will be the owner removed from
        \param mode Owner to be removed from global objects
    */
	static void removeOwner(core::Process &proc, const core::Mode &mode);

};

}}

#endif
