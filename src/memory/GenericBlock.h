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

#ifndef GMAC_MEMORY_GENERICBLOCK_H_
#define GMAC_MEMORY_GENERICBLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"

#include "util/gmac_base.h"

#include "Block.h"

namespace __impl {

namespace core {
    class address_space;
}

namespace memory {

template <typename State>
class BlockGroup;

template <typename State>
class GMAC_LOCAL GenericBlock :
    public gmac::memory::Block,
    public State,
    util::gmac_base<GenericBlock<State> > {
protected:
    BlockGroup<State> &parent_;

public:
    enum EndPoint {
        HOST,
        ACCELERATOR
    };

    typedef EndPoint Destination;
    typedef EndPoint Source;

    /**
     * Default construcutor
     *
     * \param protocol Memory coherence protocol used by the block
     * \param hostAddr Host memory address for applications to accesss the block
     * \param shadowAddr Shadow host memory mapping that is always read/write
     * \param size Size (in bytes) of the memory block
     * \param init Initial block state
     */
    GenericBlock(Protocol &protocol,
                 BlockGroup<State> &parent,
                 hostptr_t hostAddr,
                 hostptr_t shadowAddr,
                 size_t size,
                 typename State::ProtocolState init);

    /// Default destructor
    virtual ~GenericBlock();

    /**
     * Get memory block owner
     *
     * \return A reference to the owner mode of the memory block
     */
    core::address_space &owner() const;

    /**
     * Get memory block address at the accelerator
     *
     * \param current Execution mode requesting the operation
     * \param addr Address within the block
     * \return Accelerator memory address of the block
     */
    accptr_t get_device_addr(const hostptr_t addr) const;

    /**
     * Get memory block address at the accelerator
     *
     * \return Accelerator memory address of the block
     */
    accptr_t get_device_addr() const;

    gmacError_t toAccelerator(unsigned blockOff, size_t count);

    gmacError_t toAccelerator()
    {
        return toAccelerator(0, size_);
    }

    gmacError_t toHost(unsigned blockOff, size_t count);

    gmacError_t toHost()
    {
        return toHost(0, size_);
    }

    gmacError_t copyToBuffer(core::io_buffer &buffer, size_t bufferOff,
                             size_t blockOff, size_t count, Source src) const;

    gmacError_t copyFromBuffer(size_t blockOff, core::io_buffer &buffer,
                               size_t bufferOff, size_t count, Destination dst) const;

    gmacError_t copyFromBlock(size_t dstOff, GenericBlock &srcBlock,
                              size_t srcOff, size_t count,
                              Destination dst,
                              Source src) const;

    gmacError_t memset(int v, size_t count, size_t blockOffset, Destination dst) const;

#if 0
    void addOwner(core::address_space &owner, accptr_t addr);

    void removeOwner(core::address_space &owner);
#endif
};


}}

#include "GenericBlock-impl.h"

#endif
