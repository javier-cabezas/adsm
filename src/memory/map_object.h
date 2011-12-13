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

#ifndef GMAC_MEMORY_ARENA_H_
#define GMAC_MEMORY_ARENA_H_

#include <map>
#include <set>

#include "config/common.h"
#include "util/lock.h"
#include "util/NonCopyable.h"

#include "protocol/common/block_state.h"

namespace __impl {

namespace core {
class Mode;
#if 0
class Process;
#endif
namespace hpe { class AddressSpace; }
}

namespace memory {
class object;
class protocol_interface;

//! A map of objects that is not bound to any Mode
class GMAC_LOCAL map_object :
    protected gmac::util::lock_rw<map_object>,
    protected std::map<const hostptr_t, object *>,
    public util::NonCopyable {
protected:
    typedef std::map<const hostptr_t, object *> Parent;
    typedef gmac::util::lock_rw<map_object> Lock;

#if 0
    core::Process &parent_;
#endif
    protocol_interface &protocol_;

    bool modifiedObjects_;
    bool releasedObjects_;

#ifdef USE_VM
    __impl::memory::vm::Bitmap bitmap_;
#endif

#ifdef DEBUG
    static Atomic StatsInit_;
    static Atomic StatDumps_;
    static std::string StatsDir_;
    static void statsInit();
#endif

    void modifiedObjects_unlocked();

    /**
     * Find an object in the map
     *
     * \param addr Starting memory address within the object to be found
     * \param size Size (in bytes) of the memory range where the object can be
     * found
     * \return First object inside the memory range. NULL if no object is found
     */
    object *mapFind(const hostptr_t addr, size_t size) const;
public:
    /**
     * Default constructor
     *
     * \param name Name of the object map used for tracing
     */
    map_object(const char *name);

    /**
     * Default destructor
     */
    virtual ~map_object();

    /**
     * Decrements the reference count of the contained objects
     */
    void cleanUp();


    /**
     * Get the number of objects in the map
     *
     * \return Number of objects in the map
     */
    size_t size() const;

#if 0
    /**
     * Gets the parent process of the map
     *
     * \return A reference to the parent process
     */
    core::Process &getProcess();

    /**
     * Gets the parent process of the map
     *
     * \return A reference to the parent process
     */
    const core::Process &getProcess() const;
#endif

    /**
     * Insert an object in the map
     *
     * \param obj Object to insert in the map
     * \return True if the object was successfuly inserted
     */
    virtual bool addObject(object &obj);

    /**
     * Remove an object from the map
     *
     * \param obj Object to remove from the map
     * \return True if the object was successfuly removed
     */
    virtual bool removeObject(object &obj);

    /**
     * Find the firs object in a memory range
     *
     * \param addr Starting address of the memory range where the object is
     * located
     * \param size Size (in bytes) of the memory range where the object is
     * located
     * \return First object within the memory range. NULL if no object is found
     */
    virtual object *getObject(const hostptr_t addr, size_t size = 0) const;

    /**
     * Get the amount of memory consumed by all objects in the map
     *
     * \return Size (in bytes) of the memory consumed by all objects in the map
     */
    size_t memorySize() const;

    /**
     * Execute an operation on all the objects in the map
     *
     * \param f Operation to be executed
     * \sa __impl::memory::object::acquire
     * \sa __impl::memory::object::toHost
     * \sa __impl::memory::object::toAccelerator
     * \return Error code
     */
    hal::event_ptr forEachObject(hal::event_ptr (object::*f)(gmacError_t &), gmacError_t &err);

    /**
     * Execute an operation on all the objects in the map passing an argument
     * \param f Operation to be executed
     * \param p Parameter to be passed
     * \sa __impl::memory::object::removeOwner
     * \sa __impl::memory::object::realloc
     * \return Error code
     */
    template <typename T>
    hal::event_ptr forEachObject(hal::event_ptr (object::*f)(T &, gmacError_t &), T &p, gmacError_t &err);

#ifdef DEBUG
    gmacError_t dumpObjects(const std::string &dir, std::string prefix, protocol::common::Statistic stat) const;
    gmacError_t dumpObject(const std::string &dir, std::string prefix, protocol::common::Statistic stat, hostptr_t ptr) const;
#endif

    /**
     * Tells if the objects of the mode have been already invalidated
     * \return Boolean that tells if objects of the mode have been already
     * invalidated
     */
    bool hasModifiedObjects() const;

    /**
     * Notifies the mode that one (or several) of its objects have been validated
     */
    void modifiedObjects();

    /**
     * Notifies the mode that one (or several) of its objects has been invalidated
     */
    void invalidateObjects();

    /**
     * Tells if the objects of the mode have been already released to the
     * accelerator
     * \return Boolean that tells if objects of the mode have been already
     * released to the accelerator
     */
    bool releasedObjects() const;

    /**
     * Releases the ownership of the objects of the mode to the accelerator
     * and waits for pending transfers
     */
    gmacError_t releaseObjects();

    /**
     * Waits for kernel execution and acquires the ownership of the objects
     * of the mode from the accelerator
     */
    gmacError_t acquireObjects();

    /**
     * Gets a reference to the memory protocol used by the mode
     * \return A reference to the memory protocol used by the mode
     */
    protocol_interface &getProtocol();

#ifdef USE_VM
    memory::vm::Bitmap &getBitmap();
    const memory::vm::Bitmap &getBitmap() const;
#endif
};

}}

#include "map_object-impl.h"

#endif
