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

#ifndef GMAC_MEMORY_BLOCK_H_
#define GMAC_MEMORY_BLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"
#include "memory/Protocol.h"
#include "util/Reference.h"
#include "util/Lock.h"
#include "util/Logger.h"

namespace __impl { 

namespace core {
	class Mode;
	class IOBuffer;
}

namespace memory {


//! Memory block
/*!
    A memory block is a coherence unit of shared memory objects in GMAC, which are a collection of memory blocks.  Each
    memory block has an unique host memory address, used by applications to access the shared data in the CPU code, and
    a shadow host memory address used by GMAC to update the contents of the block. Upon creation, a memory block also
    has one or more accelerator memory addresses, used by the application to access the data from the accelerator, and
    owners which are those execution modes allowed to access the memory block from the accelerator. However, a memory
    block might lose its owner and memory addresses (e.g., when the execution mode owning the memory block dies) and
    stil be accessible from the CPU.

    Memory block methods should only be called from GMAC objects and GMAC memory coherence protocols.
*/
class GMAC_LOCAL Block : public gmac::util::Lock, public util::Reference {
    DBC_FORCE_TEST(Block)

protected:
    //! Memory coherence protocol used by the block
	Protocol &protocol_;

    //! Block size (in bytes)
	size_t size_;

    //! Host address where for applications to access the block. 
	uint8_t *addr_;

    //! Shadow host memory mapping that is always read/write.
	uint8_t *shadow_;

#ifdef USE_VM
    //! Last addr
    unsigned sequentialFaults_;
    unsigned faults_;
	unsigned lastSubBlock_;
    size_t subBlockSize_;

    void resetBitmapStats();
    void updateBitmapStats(const hostptr_t addr, bool write);
#endif

    //! Default construcutor
    /*!
        \param protocol Memory coherence protocol used by the block
        \param addr Host memory address for applications to accesss the block
        \param shaodw Shadow host memory mapping that is always read/write
        \param size Size (in bytes) of the memory block
    */
	Block(Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size);

    //! Default destructor
    virtual ~Block();
public:
#ifdef USE_VM
    bool isSequentialAccess() const;
    unsigned getFaults() const;
    unsigned getSequentialFaults() const;

    hostptr_t getSubBlockAddr(const hostptr_t addr) const;
    size_t getSubBlockSize() const;
    unsigned getSubBlocks() const;

    bool isSubBlockPresent(const hostptr_t addr) const;
    void setSubBlockPresent(const hostptr_t addr);
    void setBlockPresent();
#endif

    //! Host memory address where the block starts
    /*!
        \return Starting host memory address of the block
    */
    uint8_t *addr() const;

    //! Block size
    /*!
        \return Size in bytes of the memory block
    */
	size_t size() const;

    //! Signal handler for faults caused due to memory reads
    /*!
        \param addr Faulting address
        \return Error code
    */
	gmacError_t signalRead(hostptr_t addr);

    //! Signal handler for faults caused due to memory writes
    /*!
        \param addr Faulting address
        \return Error code
    */
	gmacError_t signalWrite(hostptr_t addr);

    //! Request a memory coherence operation
    /*!
        \param op Memory coherence operation to be executed
        \return Error code
    */
	gmacError_t coherenceOp(Protocol::CoherenceOp op);

    //! Request a memory operation over an I/O buffer
    /*!
        \param op Memory operation to be executed
        \param buffer IOBuffer where the operation will be executed
        \param size Size (in bytes) of the memory operation
        \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
        \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa copyToHost(core::IOBuffer &, size_t, unsigned, unsigned) const
        \sa copyToAccelerator(core::IOBuffer &, size_t, unsigned, unsigned) const
        \sa copyFromHost(core::IOBuffer &, size_t, unsigned, unsigned) const
        \sa copyFromAccelerator(core::IOBuffer &, size_t, unsigned, unsigned) const
        \sa __impl::memory::Protocol
    */
	gmacError_t memoryOp(Protocol::MemoryOp op, core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset, unsigned blockOffset);
    
    //! Initializes a memory range within the block to a specific value
    /*!
        \param v Value to initialize the memory to
        \param size Size (in bytes) of the memory region to be initialized
        \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
        \return Error code
    */
    gmacError_t memset(int v, size_t size, unsigned blockOffset = 0) const;

    //! Copy data from host memory to the memory block
    /*!
        \param src Source host memory address to copy the data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) from the begining of the block to copy the data to
        \return Error code
    */
    gmacError_t memcpyFromMemory(const hostptr_t src, size_t size, unsigned blockOffset = 0) const;

    //! Copy data from a GMAC object to the memory block
    /*!
        \param object GMAC memory object to copy data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) from the begining of the block to copy the data to
        \return Error code
    */
    gmacError_t memcpyFromObject(const Object &object, size_t size, 
        unsigned blockOffset = 0, unsigned objectOffset = 0);

    //! Copy data from the memory block to host memory
    /*!
        \param dst Destination host memory address to copy the data to
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) from the begining of the block to start copying data from
        \return Error code
    */
    gmacError_t memcpyToMemory(hostptr_t dst, size_t size, unsigned blockOffset = 0) const;

    //! Get memory block owner
    /*!
        \return Owner of the memory block
    */
	virtual core::Mode &owner() const = 0;

    //! Get memory block address at the accelerator
    /*!
        \return Accelerator memory address of the block
    */
	virtual accptr_t acceleratorAddr(const hostptr_t addr) const = 0;

    //! Ensures that the host memory has a valid and accessible copy of the data
    /*!
        \return Error code
    */
	virtual gmacError_t toHost() const = 0;

    //! Ensures that the host memory has a valid and accessible copy of the data
    /*!
        \return Error code
    */
	virtual gmacError_t toAccelerator() = 0;

    //! Copy the data from a host memory location to the block host memory
    /*!
        \param src Source host memory address to copy the data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) at the begining of the block to copy the data to
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t copyToHost(const hostptr_t src, size_t size, 
        unsigned blockOffset = 0) const = 0;

    //! Copy data from an I/O buffer to the block host memory
    /*!
        \param buffer IOBuffer where the operation will be executed
        \param size Size (in bytes) of the memory operation
        \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
        \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
        \return Error code
        \warning This method should be only called from a Protocol class
    */
	virtual gmacError_t copyToHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;

    //! Copy the data from a host memory location to the block accelerator memory
    /*!
        \param src Source host memory address to copy the data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) at the begining of the block to copy the data to
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t copyToAccelerator(const hostptr_t src, size_t size,
        unsigned blockOffset = 0) const = 0;

    //! Copy data from an I/O buffer to the block accelerator memory
    /*!
        \param buffer IOBuffer where the operation will be executed
        \param size Size (in bytes) of the memory operation
        \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
        \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
	virtual gmacError_t copyToAccelerator(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
	
    //! Copy the data from the block host memory to a host memory location
    /*!
        \param dst Destination host memory address to copy the data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) at the begining of the block to copy the data to
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t copyFromHost(hostptr_t dst, size_t size,
        unsigned blockOffset = 0) const = 0;

    //! Copy data from the block host memory to an I/O buffer
    /*!
        \param buffer IOBuffer where the operation will be executed
        \param size Size (in bytes) of the memory operation
        \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
        \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
	virtual gmacError_t copyFromHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;

    //! Copy the data from the block accelerator memory to a host memory location
    /*!
        \param dst Destination host memory address to copy the data from
        \param size Size (in bytes) of the data to be copied
        \param blockOffset Offset (in bytes) at the begining of the block to copy the data to
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t copyFromAccelerator(hostptr_t dst, size_t size,
        unsigned blockOffset = 0) const = 0;

    //! Copy data from the block accelerator memory to an I/O buffer
    /*!
        \param buffer IOBuffer where the operation will be executed
        \param size Size (in bytes) of the memory operation
        \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
        \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
	virtual gmacError_t copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const = 0;
    
    //! Initializes a memory range within the block host memory to a specific value
    /*!
        \param v Value to initialize the memory to
        \param size Size (in bytes) of the memory region to be initialized
        \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t hostMemset(int v, size_t size,
        unsigned blockOffset = 0) const = 0;

    //! Initializes a memory range within the block accelerator memory to a specific value
    /*!
        \param v Value to initialize the memory to
        \param size Size (in bytes) of the memory region to be initialized
        \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
        \return Error code
        \warning This method should be only called from a Protocol class
        \sa __impl::memory::Protocol
    */
    virtual gmacError_t acceleratorMemset(int v, size_t size, 
        unsigned blockOffset = 0) const = 0;
    
    
};


}}

#include "Block-impl.h"

#ifdef USE_DBC
#include "memory/dbc/Block.h"
#endif

#endif
