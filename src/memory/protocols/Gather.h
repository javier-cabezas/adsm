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

#ifdef USE_VM

#ifndef GMAC_MEMORY_PROTOCOL_GATHER_H_
#define GMAC_MEMORY_PROTOCOL_GATHER_H_

#include "config/common.h"
#include "include/gmac/types.h"

#include "memory/Handler.h"
#include "memory/Protocol.h"
#include "util/Lock.h"

#include "BlockList.h"

namespace __impl {

namespace core {
    class IOBuffer;
    class Mode;
}

namespace memory {
class object;
class block;
template<typename T> class StateBlock;

namespace protocol {

class GMAC_LOCAL GatherBuffer : public gmac::util::Lock {
    std::map<accptr_t, block *> subBlockList_;

public:
    GatherBuffer(unsigned subBlocks);

    bool addSubBlock(block &b, unsigned index);
    bool removeSubBlock(const block &b, unsigned index);
    void to_device();
};

class GMAC_LOCAL GatherBuffers : public gmac::util::Lock {
    GatherBuffer active_, passive_;

public:
    GatherBuffer(unsigned subBlocks);

    GatherBuffer &getActive() const;

    bool addSubBlock(block &b, unsigned index);
    bool removeSubBlock(const block &b, unsigned index);
    void release();
};


//! A lazy memory coherence protocol
/*! This protocol eagerly transfer data from host to accelerator memory if the user
    sets up a limit, otherwise data is transferred when the use requests a
    release operation. Data is transferred from accelerator memory to host memory
    lazily, whenever it is needed by the application
*/
class GMAC_LOCAL GatherBase : public protocol, handler, gmac::util::Lock {
public:
    //! Protocol states
    typedef enum {
        Invalid, /*!< Valid copy of the data in accelerator memory */
        ReadOnly, /*!< Valid copy of the data in both host and accelerator memory */
        Dirty, /*!< Valid copy of the data in host memory */
        HostOnly /*< Data only allowed in host memory */
    } State;
protected:
    //! Return the state corresponding to a memory protection
    /*!
        \param prot Memory protection
        \return Protocol state
    */
        State state(GmacProtection prot) const;

    //! Maximum number of blocks in dirty state
    size_t limit_;

    //! Dirty block list
    //! List of all memory blocks in Dirty state
    list_block dbl_;

    //! Add a new block to the Dirty Block List
    void addDirty(block &block);

    //! Default constructor
    /*!
        \param limit Maximum number of blocks in Dirty state. -1 for an infinite number
    */
    GatherBase(size_t limit);

    //! Default destructor
    virtual ~GatherBase();

public:
    // Protocol Interface
        void delete_object(object &obj);

    bool needs_update(const block &block) const;

    gmacError_t signal_read(block &block, host_ptr addr);

    gmacError_t signal_write(block &block, host_ptr addr);

    gmacError_t acquire(block &obj);

    gmacError_t acquireWithBitmap(block &obj);

    gmacError_t releaseObjects();

    gmacError_t release(block &block);

    gmacError_t map_to_device(block &block);

    gmacError_t unmap_from_device(block &block);

    gmacError_t remove_block(block &block);

        gmacError_t to_host(block &block);

    gmacError_t to_device(block &block);

        gmacError_t copyToBuffer(const block &block, core::IOBuffer &buffer, size_t size,
                size_t bufferOffset, size_t blockOffset) const;

        gmacError_t copyFromBuffer(const block &block, core::IOBuffer &buffer, size_t size,
                size_t bufferOffset, size_t blockOffset) const;

    gmacError_t memset(const block &block, int v, size_t size,
        size_t blockOffset) const;
};

template<typename T>
class GMAC_LOCAL Gather : public GatherBase {
public:
    //! Default constructor
    /*!
        \param limit Maximum number of blocks in Dirty state. -1 for an infnite number
    */
    Gather(size_t limit);

    //! Default destructor
    virtual ~Gather();

    // Protocol Interface
    memory::object *create_object(size_t size, host_ptr cpuPtr,
        GmacProtection prot, unsigned flags);
};

}}}

#include "Gather-impl.h"

#endif

#endif
