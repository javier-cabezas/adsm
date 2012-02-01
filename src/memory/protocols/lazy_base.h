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

#ifndef GMAC_MEMORY_PROTOCOL_LAZYBASE_H_
#define GMAC_MEMORY_PROTOCOL_LAZYBASE_H_

#include "config/common.h"
#include "include/gmac/types.h"

#include "memory/handler.h"
#include "memory/protocol.h"
#include "util/lock.h"

#include "common/list_block.h"
#include "lazy/block.h"
#include "lazy/lazy_types.h"

namespace __impl {

namespace memory {
class object;
class block;

template <typename State> class StateBlock;

namespace protocols {
/**
 * A lazy memory coherence protocol.
 *
 * This protocol eagerly transfer data from host to accelerator memory if the user
 * sets up a limit, otherwise data is transferred when the use requests a
 * release operation. Data is transferred from accelerator memory to host memory
 * lazily, whenever it is needed by the application
 */
class GMAC_LOCAL lazy_base :
    public protocol, handler,
    private gmac::util::mutex<lazy_base> {
    DBC_FORCE_TEST(lazy_base)
    typedef gmac::util::mutex<lazy_base> Lock;

protected:
    /** Return the state corresponding to a memory protection
     *
     * \param prot Memory protection
     * \return Protocol state
     */
    lazy_types::State state(int flagsHost, int flagsDevice) const;

    /// Uses eager update
    bool eager_;

    /// Maximum number of blocks in dirty state
    size_t limit_;

    /// Dirty block list. List of all memory blocks in Dirty state
    common::list_block dbl_;

    /// Add a new block to the Dirty Block List
    void add_dirty(lazy_types::block_ptr block);

    /** Default constructor
     *
     * \param eager Tells if protocol uses eager update
     */
    explicit lazy_base(bool eager);

    /// Default destructor
    virtual ~lazy_base();

public:
    // Protocol Interface
    void delete_object(object &obj);

    bool needs_update(common::block_const_ptr block) const;

    TESTABLE hal::event_ptr signal_read(common::block_ptr block, host_ptr addr, gmacError_t &err);

    TESTABLE hal::event_ptr signal_write(common::block_ptr block, host_ptr addr, gmacError_t &err);

    TESTABLE hal::event_ptr acquire(common::block_ptr block, GmacProtection prot, gmacError_t &err);

    TESTABLE hal::event_ptr release(common::block_ptr block, gmacError_t &err);

#ifdef USE_VM
    hal::event_ptr acquireWithBitmap(common::block_ptr block, gmacError_t &err);
#endif

    TESTABLE hal::event_ptr release_all(gmacError_t &err);
    //gmacError_t releasedAll();

    TESTABLE hal::event_ptr map_to_device(common::block_ptr block, gmacError_t &err);

    TESTABLE hal::event_ptr unmap_from_device(common::block_ptr block, gmacError_t &err);

    TESTABLE hal::event_ptr remove_block(common::block_ptr block, gmacError_t &err);

    TESTABLE hal::event_ptr to_host(common::block_ptr block, gmacError_t &err);

#if 0
    hal::event_ptr to_device(common::block_ptr block, gmacError_t &err);

    TESTABLE gmacError_t copyToBuffer(common::block_ptr block, core::io_buffer &buffer, size_t size,
                                      size_t bufferOffset, size_t blockOffset);

    TESTABLE gmacError_t copyFromBuffer(common::block_ptr block, core::io_buffer &buffer, size_t size,
                                        size_t bufferOffset, size_t blockOffset);
#endif

    TESTABLE hal::event_ptr memset(common::block_ptr block, size_t blockOffset, int v, size_t size,
                                   gmacError_t &err);

    TESTABLE hal::event_ptr flush_dirty(gmacError_t &err);

    //bool isInAccelerator(block_ptr block);
    hal::event_ptr copy_block_to_block(common::block_ptr d, size_t dstOffset,
                                       common::block_ptr s, size_t srcOffset, size_t count, gmacError_t &err);

    hal::event_ptr copy_to_block(common::block_ptr dst, size_t dstOffset,
                                 host_const_ptr src,
                                 size_t count, gmacError_t &err);

    hal::event_ptr copy_from_block(host_ptr dst,
                                   common::block_ptr src, size_t srcOffset,
                                   size_t count, gmacError_t &err);

    hal::event_ptr to_io_device(hal::device_output &output,
                                common::block_ptr src, size_t srcffset,
                                size_t count, gmacError_t &err);

    hal::event_ptr from_io_device(common::block_ptr dst, size_t dstOffset,
                                  hal::device_input &input,
                                  size_t count, gmacError_t &err);

    gmacError_t dump(common::block_ptr block, std::ostream &out, common::Statistic stat);
};

}}}

#ifdef USE_DBC
#include "dbc/lazy_base.h"
#endif

#endif
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
