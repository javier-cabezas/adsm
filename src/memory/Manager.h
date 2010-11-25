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

#ifndef GMAC_MEMORY_MANAGER_H_
#define GMAC_MEMORY_MANAGER_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/Singleton.h"

namespace __impl {

namespace core {
class IOBuffer;
class Mode;
}

namespace memory {

class Object;
class Protocol;

//! Memory Manager Interface

//! Memory Managers orchestate the data transfers between host and device memories
class GMAC_LOCAL Manager : public __impl::util::Singleton<gmac::memory::Manager> {
    DBC_FORCE_TEST(Manager)

	// Needed to let Singleton call the protected constructor
	friend class util::Singleton<Manager>;
private:
#ifdef USE_VM
    void checkBitmapToHost();
    void checkBitmapToDevice();
#endif
protected:
    //! Allocated a host mapped memory
    /*!
        \param Pointer to the variable that will store the begining of the allocated memory
        \param Size (in bytes) of the memory to be allocated
        \return Error core
    */
    gmacError_t hostMappedAlloc(void **addr, size_t size);

    //! Copy data from host memory to an object
    /*!
        \param obj Destination object
        \param src Source host memory address
        \param size Size (in bytes) of the data to be copied
        \param objOffset Offset (in bytes) from the begining of the object to copy the data to
        \return Error code
    */
    gmacError_t memcpyToObject(const Object &obj, const void *src, 
        size_t size, unsigned objOffset);

    //! Copy data from object to object
    /*!
        \param dstObj Destination object
        \param srcObj Source object
        \param size Size (in bytes) of the data to be copied
        \param dstOffset Offset (in bytes) from the begining of the destination object to copy data to
        \param srcOffset Offset (in bytes) from the begining og the source object to copy data from
        \return Error code
    */
    gmacError_t memcpyToObject(const Object &dstObj, const Object &srcObj,
        size_t size, unsigned dstOffset, unsigned srcOffset);

    //! Copy data from an object to host memory
    /*!
        \param dst Destination object
        \param src Source host memory address
        \param size Size (in bytes) of the data to be copied
        \param objOffset Offset (in bytes) from the begining of the destination object to copy data to
        \return Error code
    */
    gmacError_t memcpyFromObject(void *dst, const Object &obj, size_t size,
        unsigned objOffset);

    //! Default constructor
    Manager();

    //! Default destructor
    virtual ~Manager();
public:
    //TESTABLE gmacError_t map(void *addr, size_t size, GmacProtection prot);
    //TESTABLE gmacError_t unmap(void *addr, size_t size);

    //! Allocate private shared memory 
    //! Memory allocated with this call is only accessible by the accelerator associated to 
    //! the execution thread requesting the allocation
    /*!
        \param addr Memory address of a pointer to store the host address of the allocated memory
        \param size Size (in bytes) of shared memory to be allocated
        \return Error code
    */         
    TESTABLE gmacError_t alloc(void **addr, size_t size);

    //! Allocate public shared read-only memory
    //! Memory allocated with this call is accessible (read-only) from any accelerator
    /*!
        \param addr Memory address of a pointer to store the host address of the allocated memory
        \param size Size (in bytes) of shared memory to be allocated
        \param hint Type of memory (distributed or hostmapped) to be allocated
        \return Error code
    */
    TESTABLE gmacError_t globalAlloc(void **addr, size_t size, GmacGlobalMallocType hint);

    //! Release shared memory
    /*!
        \param addr Memory address of the shared memory chunk to be released
        \return Error code
    */
    TESTABLE gmacError_t free(void *addr);

    //! Get the device address associated to a shared memory address
    /*!
        \param addr Host shared memory address
        \param Accelerator memory address
    */
    TESTABLE void *translate(const void *addr);

    //! Get the CPU ownership of all objects bound to the current execution mode
    /*!
        \return Error code
    */
    gmacError_t acquireObjects();

    //! Release the CPU ownership of all objects bound to the current execution mode
    /*!
        \return Error code
    */
    gmacError_t releaseObjects();

    //! Invalidate the host memory of all objects bound to the current execution mode
    /*!
        \return Error code
    */
    gmacError_t invalidate();

    //! Notify a memory fault caused by a load operation
    /*!
        \param addr Host memory address causing the memory fault
        \return True if the Manager was able to fix the fault condition
    */
    TESTABLE bool read(void *addr);

    //! Notify a memory fault caused by a store operation
    /*!
        \param addr Host memory address causing the memory fault
        \return True if the Manager was able to fix the fault condition
    */
    TESTABLE bool write(void *addr);

    //! Copy data from a memory address of a memory object to an I/O buffer
    /*!
        \param buffer Destionation I/O buffer to copy the data to
        \param addr Host memory address corresponding to a memory object to copy data from
        \param size Size (in bytes) of the data to be copied
        \return Error code
    */
	TESTABLE gmacError_t toIOBuffer(__impl::core::IOBuffer &buffer, const void *addr, size_t size);

    //! Copy data from an I/O buffer to a host memory address of a memory object
    /*!
        \param buffer Source I/O buffer to copy the data from
        \param addr Host memory address corresponding to a memory object to copy data to
        \param size Size (in bytes) of the data to be copied
        \return Error code
    */
	TESTABLE gmacError_t fromIOBuffer(void *addr, __impl::core::IOBuffer &buffer, size_t size);

    //! Initialize to a given value the contents of a host address of a memory object
    /*!
        \param dst Host memory address corresponding to a memory object to set the memory contents
        \param c Value used to initialize memory
        \param size Size (in bytes) of the memory to initialize
        \return Error code
    */
    TESTABLE gmacError_t memset(void *dst, int c, size_t size);

    //! Copy data from and/or to host memory addresses of memory objects
    /*!
        \param dst Destination host memory addrees of a memory objec to copy the data to
        \param src Source host memory address that might or might not correspond to a memory object
        \param size Size (in bytes) of the amoun of data to be copied
    */
    TESTABLE gmacError_t memcpy(void *dst, const void *src, size_t size);

#if 0
    gmacError_t moveTo(void *addr, __impl::core::Mode &mode);
#endif

};

}}

#include "Manager-impl.h"

#ifdef USE_DBC
#include "memory/dbc/Manager.h"
#endif

#endif
