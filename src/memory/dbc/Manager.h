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

#ifndef GMAC_MEMORY_DBC_MANAGER_H_
#define GMAC_MEMORY_DBC_MANAGER_H_

#include "dbc/types.h"
#include "dbc/Contract.h"
#include "memory/Manager.h"

namespace __dbc { namespace memory {

//! Memory Manager Interface

//! Memory Managers implement a policy to move data from/to
//! the CPU memory to/from the accelerator memory.
class GMAC_LOCAL Manager :
    public __impl::memory::Manager,
    public virtual Contract {
    DBC_TESTED(__impl::memory::Manager)

protected:

    /**
     * Default destructor
     */
    ~Manager();
public:
    /**
     * Default constructor
     */
    Manager(__impl::core::Process &proc);

    /**
     * Allocate private shared memory.
     * Memory allocated with this call is only accessible by the accelerator
     * associated to the execution thread requesting the allocation
     * \param mode Execution mode requesing the allocation
     * \param addr Memory address of a pointer to store the host address of the
     * allocated memory
     * \param size Size (in bytes) of shared memory to be allocated
     * \return Error code
     */
    gmacError_t alloc(__impl::core::Mode &mode, hostptr_t *addr, size_t size);

    /**
     * Release shared memory
     * \param mode Execution mode requesting the memory release
     * \param addr Memory address of the shared memory chunk to be released
     * \return Error code
     */
    gmacError_t free(__impl::core::Mode &mode, hostptr_t addr);

    /**
     * Notify a memory fault caused by a load operation
     * \param mode Execution mode causing the fault
     * \param addr Host memory address causing the memory fault
     * \return True if the Manager was able to fix the fault condition
     */
    bool read(__impl::core::Mode &mode, hostptr_t addr);

    /**
     * Notify a memory fault caused by a store operation
     * \param mode Execution mode causing the fault
     * \param addr Host memory address causing the memory fault
     * \return True if the Manager was able to fix the fault condition
     */
    bool write(__impl::core::Mode &mode, hostptr_t addr);

    /**
     * Copy data from a memory object to an I/O buffer
     * \param mode Execution mode requesting the data transfer
     * \param buffer Destionation I/O buffer to copy the data to
     * \param bufferOff Offset within the buffer to copy data to
     * \param addr Host memory address corresponding to a memory object to copy
     * data from
     * \param size Size (in bytes) of the data to be copied
     * \return Error code
     */
    gmacError_t toIOBuffer(__impl::core::Mode &mode, __impl::core::IOBuffer &buffer, size_t bufferOff, const hostptr_t addr, size_t size);


    /**
     * Copy data from an I/O buffer to a memory object
     * \param mode Execution mode requesting the data transfer
     * \param addr Host memory address corresponding to a memory object to copy
     * data to
     * \param buffer Source I/O buffer to copy the data from
     * \param bufferOff Offset within the buffer to copy data from
     * \param size Size (in bytes) of the data to be copied
     * \return Error code
     */
    gmacError_t fromIOBuffer(__impl::core::Mode &mode, hostptr_t addr, __impl::core::IOBuffer &buffer, size_t bufferOff, size_t size);
};

}}

#endif
