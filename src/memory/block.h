/* Copyright (c) 2009, 2010, 2011 University of Illinois
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
#include "util/lock.h"
#include "util/Logger.h"
#include "util/Reference.h"

#include "util/UniquePtr.h"

/** Description for __impl. */
namespace __impl {

namespace core {
    class address_space;

    typedef util::shared_ptr<address_space> address_space_ptr;
    typedef util::shared_ptr<const address_space> address_space_const_ptr;
}

namespace memory {

class block;

typedef __impl::util::shared_ptr<block> block_ptr;

/** Memory block
 * A memory block is a coherence unit of shared memory objects in GMAC, which are a collection of memory blocks.  Each
 * memory block has an unique host memory address, used by applications to access the shared data in the CPU code, and
 * a shadow host memory address used by GMAC to update the contents of the block. Upon creation, a memory block also
 * has one or more accelerator memory addresses, used by the application to access the data from the accelerator, and
 * owners which are those execution modes allowed to access the memory block from the accelerator. However, a memory
 * block might lose its owner and memory addresses (e.g., when the execution mode owning the memory block dies) and
 * stil be accessible from the CPU.
 * Memory block methods should only be called from GMAC objects and GMAC memory coherence protocols.
 */
class GMAC_LOCAL block :
	public gmac::util::mutex<block> {
    DBC_FORCE_TEST(block)

	typedef gmac::util::mutex<block> Lock;

    friend class object;

protected:
    /** Block size (in bytes) */
    size_t size_;

    /** Host address where for applications to access the block. */
    hostptr_t addr_;

    /** Shadow host memory mapping that is always read/write. */
    hostptr_t shadow_;

    /**
     * Default construcutor
     *
     * \param addr Host memory address for applications to accesss the block
     * \param shadow Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     */
    block(hostptr_t addr, hostptr_t shadow, size_t size);

    /**
     * Default destructor
     */
    virtual ~block();

public:

    /**
     * Host memory address where the block starts
     * \return Starting host memory address of the block
     */
    hostptr_t addr() const;

    /**
     *  Block size
     * \return Size in bytes of the memory block
     */
    size_t size() const;

protected:
    /**
     * Signal handler for faults caused due to memory reads
     * \param addr Faulting address
     * \return Error code
     */
    gmacError_t signal_read(hostptr_t addr);

    /**
     * Signal handler for faults caused due to memory writes
     * \param addr Faulting address
     * \return Error code
     */
    gmacError_t signal_write(hostptr_t addr);

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \return Error code
     */
    gmacError_t toAccelerator() { return toAccelerator(0, size_); }
    virtual gmacError_t toAccelerator(unsigned blockOff, size_t count) = 0;

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \return Error code
     */
    gmacError_t toHost() { return toHost(0, size_); }

    /**
     * Ensures that the host memory has a valid and accessible copy of the data
     * \param blockOff Offset within the block
     * \param count Size (in bytes)
     * \return Error code
     */
    virtual gmacError_t toHost(unsigned blockOff, size_t count) = 0;

public:
#if 0
    /**
     * Initializes a memory range within the block to a specific value
     * \param v Value to initialize the memory to
     * \param size Size (in bytes) of the memory region to be initialized
     * \param blockOffset Offset (in bytes) from the begining of the block to perform the initialization
     * \return Error code
     */
    TESTABLE gmacError_t memset(int v, size_t size, size_t blockOffset);

    /** Request a memory coherence operation
     *
     *  \param op Memory coherence operation to be executed
     *  \return Error code
     */
    template <typename R>
    R coherenceOp(R (protocol_interface::*op)(block_ptr));
    template <typename R, typename T>
    R coherenceOp(R (protocol_interface::*op)(block_ptr, T &), T &param);

    static hal::event_t copy_op(protocol_interface::CopyOp1To op, block_ptr dst, size_t dstOff, const hostptr_t src, size_t count, gmacError_t &err);
    static hal::event_t copy_op(protocol_interface::CopyOp1From op, hostptr_t dst, block_ptr src, size_t srcOff, size_t count, gmacError_t &err);
    static hal::event_t copy_op(protocol_interface::CopyOp2 op, block_ptr dst, size_t dstOff, block_ptr src, size_t srcOff, size_t count, gmacError_t &err);

    static hal::event_t device_op(protocol_interface::DeviceOpTo op, hal::device_output &output, block_ptr src, size_t srcOff, size_t count, gmacError_t &err);
    static hal::event_t device_op(protocol_interface::DeviceOpFrom op, block_ptr dst, size_t dstOff, hal::device_input &input, size_t count, gmacError_t &err);

#if 0
    /**
     *  Request a memory operation over an I/O buffer
     * \param op Memory operation to be executed
     * \param buffer IOBuffer where the operation will be executed
     * \param size Size (in bytes) of the memory operation
     * \param bufferOffset Offset (in bytes) from the starting of the I/O buffer where the memory operation starts
     * \param blockOffset Offset (in bytes) from the starting of the block where the memory opration starts
     * \return Error code
     * \warning This method should be only called from a Protocol class
     * \sa copyToHost(core::io_buffer &, size_t, size_t, size_t) const
     * \sa copyToAccelerator(core::io_buffer &, size_t, size_t, size_t) const
     * \sa copyFromHost(core::io_buffer &, size_t, size_t, size_t) const
     * \sa copyFromAccelerator(core::io_buffer &, size_t, size_t, size_t) const
     * \sa __impl::memory::Protocol
     */
    TESTABLE gmacError_t memoryOp(protocol_interface::MemoryOp op,
                                  core::io_buffer &buffer, size_t size, size_t bufferOffset, size_t blockOffset);
#endif

    /**
     * Copy data from a GMAC object to the memory block
     *
     * \param obj GMAC memory object to copy data from
     * \param size Size (in bytes) of the data to be copied
     * \param blockOffset Offset (in bytes) from the begining of the block to
     * copy the data to
     * \param objectOffset Offset (in bytes) from the begining of the object to
     * copy the data from
     * \return Error code
     */
    gmacError_t memcpyFromObject(const object &obj, size_t size,
                                 size_t blockOffset = 0, size_t objectOffset = 0);
#endif
    /**
     * Get memory block owner
     * \return Owner of the memory block
     */
    virtual core::address_space_ptr owner() const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t get_device_addr(const hostptr_t addr) const = 0;

    /**
     * Get memory block address at the accelerator
     * \return Accelerator memory address of the block
     */
    virtual accptr_t get_device_addr() const = 0;

    /**
     * Dump statistics about the memory block
     * \param param Stream to dump the statistics to
     * \param stat Statistic to be dumped
     * \return Error code
     */
    gmacError_t dump(std::ostream &param, protocol::common::Statistic stat);

    hostptr_t get_shadow() const;
};

}}

#include "block-impl.h"

#ifdef USE_DBC
#include "memory/dbc/block.h"
#endif

#endif
