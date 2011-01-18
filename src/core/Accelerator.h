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
 WITH THE SOFTWARE.
 */

#ifndef GMAC_CORE_ACCELERATOR_H_
#define GMAC_CORE_ACCELERATOR_H_

#include <stddef.h>

#include <set>

#include "config/common.h"
#include "include/gmac/types.h"


namespace __impl { namespace core {

class KernelLaunch;
class Mode;
class Process;

/** Generic Accelerator Class Defines the standard interface all accelerators MUST implement */
class GMAC_LOCAL Accelerator {
protected:
    /** Identifier of the accelerator */
    unsigned id_;

    /** Identifier of the bus where the accelerator is located */
    unsigned busId_;

    /** Identifier of the accelerator within the bus where the accelerator is located */
    unsigned busAccId_;

    /** Value that tells if the accelerator is integrated and therefore shares
     * the physical memory with the CPU */
    bool integrated_;

    /** Value that represents the load of the accelerator */
    unsigned load_;

    /** Set of modes that run on the accelerator */
    std::set<Mode *> queue_;
public:
    /**
     * Constructs an Accelerator and initializes its information fields
     * \param n Identifier of the accelerator, must be unique in the system
     */
    Accelerator(int n);

    /**
     * Releases the generic (non-API dependant) resources of the accelerator
     */
    virtual ~Accelerator();

    /**
     * Gets the identifier of the accelerator
     * @return The identifier of the accelerator
     */
    unsigned id() const;

    /**
     * Creates and returns a mode on the given process and registers it to run
     * on the accelerator
     * \param proc Reference to a process which the mode will belong to
     * \return A pointer to the created mode or NULL if there has been an error
     */
    virtual gmac::core::Mode *createMode(Process &proc) = 0;

    /**
     * Registers a mode to be run on the accelerator. The mode must not be
     * already registered in the accelerator
     * \param mode A reference to the mode to be registered
     */
    void registerMode(Mode &mode);

    /**
     * Unegisters a mode from the accelerator. The mode must be already
     * registered in the accelerator
     * \param mode A reference to the mode to be unregistered
     */
    void unregisterMode(Mode &mode);

    /**
     * Returns a value that indicates the load of the accelerator
     * \return A value that indicates the load of the accelerator
     */
    virtual unsigned load() const;

    /**
     * Allocates memory on the accelerator memory
     * \param addr Reference to a pointer to store the address of the allocated
     * memory
     * \param size The number of bytes of the allocation
     * \param align The alignment of the allocation. This value must be a power
     * of two
     * \return Error code
     */
    virtual gmacError_t malloc(accptr_t &addr, size_t size, unsigned align = 1) = 0;

    /**
     * Releases memory previously allocated by malloc
     * \param addr A pointer with the address of the allocation to be freed
     * \return Error code
     */
    virtual gmacError_t free(accptr_t addr) = 0;

    /**
     * Waits for kernel execution and returns the execution return value
     * \return Error code
     */
    virtual gmacError_t sync() = 0;

    /**
     * Copies data from host memory to accelerator memory
     * \param acc Destination pointer to accelerator memory
     * \param host Source pointer to host memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t
        copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size) = 0;

    /**
     * Copies data from accelerator memory to host memory
     * \param host Destination pointer to host memory
     * \param acc Source pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size) = 0;

    /**
     * Copies data from accelerator memory to accelerator memory
     * \param dst Destination pointer to accelerator memory
     * \param src Source pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    virtual gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size) = 0;

    /**
     * Gets the memory information for the accelerator
     * \param free A reference to the variable where to store the amount of free
     * memory in the accelerator
     * \param total A reference to the variable where to store the total amount
     * of memory of the accelerator
     */
    virtual void memInfo(size_t &free, size_t &total) const = 0;

    // TODO: use this methods for something useful
    /**
     * Gets the bus identifier where the accelerator is located
     * \return The bus identifier where the accelerator is located
     */
    unsigned busId() const;

    /**
     * Gets the accelerator ID within the bus where the accelerator is located
     * \return The bus identifier where the accelerator is located
     */
    unsigned busAccId() const;

    /**
     * Tells if the accelerator is integrated and therefore shares the physical
     * memory with the CPU
     * \return A boolean that tells if the accelerator is integrated and
     * therefore shares the physical memory with the CPU
     */
    bool integrated() const;
};

}}

#include "Accelerator-impl.h"

#endif
