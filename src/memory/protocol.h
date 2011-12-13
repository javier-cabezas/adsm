/* Copyright (c) 2009-2011 University of Illinois
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

#ifndef GMAC_MEMORY_PROTOCOL_H_
#define GMAC_MEMORY_PROTOCOL_H_

#include <fstream>
#include <list>

#include "config/common.h"
#include "hal/types.h"
#include "include/gmac/types.h"

#include "memory/protocol/common/block_state.h"

namespace __impl {

namespace memory {

class block;
typedef __impl::util::shared_ptr<block> block_ptr;
class object;

/**
 * Base class that defines the operations to be implemented by any protocol
 */
class GMAC_LOCAL protocol {
public:
    /// Default destructor
    virtual ~protocol();

    /** Creates a new object that will be manged by this protocol
     *
     * \param size Size (in bytes) of the new object
     * \param cpuPtr Host address where the object will be create. NULL to let
     * the protocol choose
     * \param prot Memory protection for the object host memory after it is
     * created
     * \param flags Protocool specific flags
     * \return Pointer to the created object
     */
    virtual object *create_object(size_t size, hostptr_t cpuPtr,
                                  GmacProtection prot, unsigned flags) = 0;

    /**
     * Deletes an object created by this protocol
     *
     * \param obj Object to be deleted
     */
    virtual void delete_object(object &obj) = 0;

    /**
     * Checks if a memory block has an updated copy of the data in the
     * accelerator memory
     *
     * \param block Memory block to check
     * \return Whether the accelerator memory of the block has an updated copy
     * of the data or not
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual bool needs_update(const block_ptr block) const = 0;

    /**
     * Signal handler for faults caused due to memory reads
     *
     * \param block Memory block where the fault was triggered
     * \param addr  Faulting address
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr signal_read(block_ptr block, hostptr_t addr, gmacError_t &err) = 0;

    /**
     * Signal handler for faults caused due to memory writes
     *
     * \param block Memory block where the fault was triggered
     * \param addr  Faulting address
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr signal_write(block_ptr block, hostptr_t addr, gmacError_t &err) = 0;

    /** Acquires the ownership of a memory block for the CPU
     *
     * \param block Memory block whose ownership is acquired
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr acquire(block_ptr block, GmacProtection &prot, gmacError_t &err) = 0;
#ifdef USE_VM
    virtual hal::event_ptr acquireWithBitmap(block_ptr block, gmacError_t &err) = 0;
#endif

    /**
     * Releases the CPU ownership of all objects belonging to this protocol
     *
     * \return Error code
     */
    virtual hal::event_ptr release_all(gmacError_t &err) = 0;
    //virtual hal::event_ptr released_all(gmacError_t &err) = 0;

    //virtual gmacError_t release_objects(const std::list<object *> &objects) = 0;

    /**
     * Releases the CPU ownership of a memory block belonging to this protocol
     *
     * \param block Memory block whose ownership is release
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr release(block_ptr block, gmacError_t &err) = 0;

    /**
     * Removes a block from the coherence domain.
     *
     * This method ensures that the block host memory contains an updated copy
     * of the data, and then marks the block to not use the accelerator memory
     * any more. After calling this method a memory block will always remain in
     * host memory
     * \param block Memory block to remove from the coherence domain
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr unmap_from_device(block_ptr block, gmacError_t &err) = 0;

    /**
     * Adds a block to the coherence domain.
     *
     * This method marks the block to use accelerator memory. After calling
     * this method the block coherency is managed using this protocol
     * \param block Memory block to add to the coherence domain
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr map_to_device(block_ptr block, gmacError_t &err) = 0;

    /**
     * Deletes all references to the block within the protocol
     *
     * This method is used when a memory block is being destroyed to ensure that
     * the protocol does not keep any reference to the block being destroyed
     * \param block Memory block being destroyed
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr remove_block(block_ptr block, gmacError_t &err) = 0;

    /**
     * Ensures that the host memory of a block contains an updated copy of the
     * data
     *
     * \param block Memory block whose host memory is being updated
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr to_host(block_ptr block, gmacError_t &err) = 0;

#if 0
    /**
     * Ensures that the accelerator memory of a block contains an updated copy
     * of the data
     *
     * \param block Memory block whose accelerator memory is being updated
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr to_device(block_ptr block, gmacError_t &err) = 0;
#endif

    /**
     * Initializes a memory range within a memory block to a specific value
     *
     * \param block Memory block to be initialized
     * \param v Value to initialize the memory to
     * \param size Size (in bytes) of the memory region to be initialized
     * \param blockOffset Offset (in bytes) from the begining of the block to
     * perform the initialization
     * \return Error code
     * \warning This method assumes that the block is not modified during its
     * execution
     */
    virtual hal::event_ptr memset(block_ptr block, size_t blockOffset, int v, size_t size,
                                  gmacError_t &err) = 0;

    virtual hal::event_ptr flush_dirty(gmacError_t &err) = 0;

    /**
     * Copies between two memory blocks, assuming that direct copies are
     * possible regardless the location of the data (accelerator or host memory)
     *
     * \param dst Destination block
     * \param dstOffset Offset within the destination block
     * \param src Source block
     * \param srcOffset Offset within the source block
     * \param count Size (in bytes) of the memory transfer
     */
    virtual hal::event_ptr copy_block_to_block(block_ptr dst, size_t dstOffset,
    		                                   block_ptr src, size_t srcOffset,
                                               size_t count, gmacError_t &err) = 0;

    virtual hal::event_ptr copy_to_block(block_ptr dst, size_t dstOffset,
                                         hostptr_t src,
                                         size_t count, gmacError_t &err) = 0;

    virtual hal::event_ptr copy_from_block(hostptr_t dst,
    		                               block_ptr src, size_t srcOffset,
                                           size_t count, gmacError_t &err) = 0;

    virtual hal::event_ptr to_io_device(hal::device_output &output,
    		                            block_ptr src, size_t srcffset,
                                        size_t count, gmacError_t &err) = 0;

    virtual hal::event_ptr from_io_device(block_ptr dst, size_t dstOffset,
                                          hal::device_input &input,
                                          size_t count, gmacError_t &err) = 0;

    virtual gmacError_t dump(block_ptr block, std::ostream &out, protocols::common::Statistic stat) = 0;

    typedef hal::event_ptr (protocol::*CoherenceOp)(block_ptr, gmacError_t &err);

    typedef hal::event_ptr (protocol::*CopyOp1To)(block_ptr, size_t, const hostptr_t, size_t, gmacError_t &);
    typedef hal::event_ptr (protocol::*CopyOp1From)(hostptr_t, block_ptr, size_t, size_t, gmacError_t &);
    typedef hal::event_ptr (protocol::*CopyOp2)(block_ptr, size_t, block_ptr, size_t, size_t, gmacError_t &);

    typedef hal::event_ptr (protocol::*DeviceOpTo)(hal::device_output &, block_ptr, size_t, size_t, gmacError_t &);
    typedef hal::event_ptr (protocol::*DeviceOpFrom)(block_ptr, size_t, hal::device_input &, size_t, gmacError_t &);
};

typedef std::list<object *> ListObject;
extern ListObject AllObjects;

}}

#endif
