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

#ifndef GMAC_CORE_HPE_CONTEXT_H_
#define GMAC_CORE_HPE_CONTEXT_H_

#include "config/common.h"

#include "hal/types.h"

#include "include/gmac/types.h"

#include "util/lock.h"
#include "util/NonCopyable.h"
#include "util/Private.h"

namespace __impl { namespace core {
class io_buffer;

namespace hpe {
class Accelerator;
class Mode;
class Kernel;
class KernelLaunch;

/**
 * Generic context class that represents the per-thread operations' state in the accelerator
 */
class GMAC_LOCAL context : public gmac::util::lock_rw,
                           public util::NonCopyable {
    friend class resource_manager;
#if 0
    DBC_FORCE_TEST(context)
#endif
protected:
    /** Hardware ctx associated to the thread context */
    hal::context_t &ctx_;
#if 0
    /** Execution mode owning the context */
    Mode &mode_;
#endif

    /** Accelerator streams used to enqueue operations */
    hal::stream_t &streamLaunch_;
    hal::stream_t &streamToAccelerator_;
    hal::stream_t &streamToHost_;
    hal::stream_t &streamAccelerator_;

    /** I/O buffers used by the context for data transfers */
    io_buffer *bufferWrite_;
    io_buffer *bufferRead_;

    /**
     * Constructs a context for the calling thread on the given mode
     *
     * \param dev Reference to the the context
     * \param streamLaunch Command execution stream to be used to launch kernels
     * \param streamToAccelerator Command execution stream to be used to perform
     * host-to-acc transfers
     * \param streamToHost Command execution stream to be used to perform
     * acc-to-host transfers
     * \param streamAccelerator Command execution stream to be used to perform
     * acc-to-acc transfers
     */
    context(hal::context_t &ctx,
            hal::stream_t &streamLaunch,
            hal::stream_t &streamToAccelerator,
            hal::stream_t &streamToHost,
            hal::stream_t &streamAccelerator);

public:
    /**
     * Destroys the resources used by the context
     */
    virtual ~context();

    /**
     * Initialization method called on library initialization
     */
    static void init();

#if 0
    /**
     * Copies size bytes from host memory to accelerator memory
     *
     * \param acc Destination pointer to accelerator memory
     * \param host Source pointer to host memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);
    /**
     * Copies size bytes from accelerator memory to host memory
     *
     * \param host Destination pointer to host memory
     * \param acc Source pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    TESTABLE gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);
#endif

    hal::event_t *copy(accptr_t acc, const hostptr_t host, size_t count, gmacError_t &err);
    hal::event_t *copy(hostptr_t host, const accptr_t acc, size_t count, gmacError_t &err);
    hal::event_t *copy(accptr_t dst, const accptr_t src, size_t count, gmacError_t &err);

    hal::async_event_t *copy_async(accptr_t acc, const hostptr_t host, size_t count, gmacError_t &err);
    hal::async_event_t *copy_async(hostptr_t host, const accptr_t acc, size_t count, gmacError_t &err);
    hal::async_event_t *copy_async(accptr_t dst, const accptr_t src, size_t count, gmacError_t &err);

    gmacError_t copy(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count);
    gmacError_t copy(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count);

    gmacError_t copy_async(accptr_t dst, core::io_buffer &buffer, size_t off, size_t count);
    gmacError_t copy_async(core::io_buffer &buffer, size_t off, const accptr_t src, size_t count);

    hal::event_t *memset(accptr_t addr, int c, size_t count, gmacError_t &err);
    hal::async_event_t *memset_async(accptr_t addr, int c, size_t count, gmacError_t &err);

};

}}}

#ifdef USE_DBC
#if 0
#include "core/hpe/dbc/context.h"
#endif
#endif

#endif
