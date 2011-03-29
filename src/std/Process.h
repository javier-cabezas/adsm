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

#ifndef GMAC_STD_PROCESS_H_
#define GMAC_STD_PROCESS_H_

#include "config/common.h"
#include "config/order.h"
#include "include/gmac/types.h"


namespace __impl { namespace std {


/** Represents the resources used by a running process */
class GMAC_LOCAL Process {
protected:
public:
    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~Process();

    /**
     * Registers a global object in the process
     *
     * \param object Reference to the object to be registered
     * \return Error code
     */
    virtual gmacError_t globalMalloc(memory::Object &object) = 0;

    /**
     * Unregisters a global object from the process
     *
     * \param object Reference to the object to be unregistered
     * \return Error code
     */
    virtual gmacError_t globalFree(memory::Object &object) = 0;

    /**
     * Translates a host address to an accelerator address
     *
     * \param addr Host address to be translated
     * \return Accelerator address
     */
    virtual accptr_t translate(const hostptr_t addr) = 0;

    /**
     * Gets the protocol used by the process for the global objects
     *
     * \return A reference to the protocol used by the process for the global
     * objects
     */
    memory::Protocol &protocol();

    /**
     * Gets the object map that contains all the shared objects allocated in the
     * process
     *
     * \return A reference to the object map that contains all the shared
     * objects allocated in the process
     */
    memory::ObjectMap &shared();

    /**
     * Gets the object map that contains all the shared objects allocated in the
     * process
     *
     * \return A constant reference to the object map that contains all the shared
     * objects allocated in the process
     */
    const memory::ObjectMap &shared() const;

    /**
     * Gets the object map that contains all the global objects allocated in the
     * process
     *
     * \return A reference to the object map that contains all the global
     * objects allocated in the process
     */
    memory::ObjectMap &global();

    /**
     * Gets the object map that contains all the global objects allocated in the
     * process
     *
     * \return A constant reference to the object map that contains all the
     * global objects allocated in the process
     */
    const memory::ObjectMap &global() const;

    /**
     * Gets the object map that contains all the objects that have been orphaned
     * in the process
     *
     * \return A reference to the object map that contains all the objects that
     * have been orphaned in the process
     */
    memory::ObjectMap &orphans();

    /**
     * Gets the object map that contains all the objects that have been orphaned
     * in the process
     *
     * \return A constant reference to the object map that contains all the objects that
     * have been orphaned in the process
     */
    const memory::ObjectMap &orphans() const;

    /**
     * Returns the owner of the object with the smallest address within the
     * given memory range
     *
     * \param addr Starting address of the range
     * \param size Size of the range
     * \return The owner of the object with the smallest address within the
     * given memory range
     */
    Mode *owner(const hostptr_t addr, size_t size = 0) const;
};

}}

#include "Process-impl.h"

#ifdef USE_DBC
#include "core/dbc/Process.h"
#endif

#endif
