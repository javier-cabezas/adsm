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

#ifndef GMAC_CORE_HPE_CONTEXT_H_
#define GMAC_CORE_HPE_CONTEXT_H_

#include "config/common.h"
#include "include/gmac/types.h"
#include "util/Lock.h"
#include "util/NonCopyable.h"
#include "util/Private.h"

namespace __impl { namespace core { 

namespace hpe {
class Accelerator;
class Mode;
class Kernel;
class KernelLaunch;

/**
 * Generic Context Class.
 * Represents the per-thread operations' state in the accelerator
 */
class GMAC_LOCAL Context : public gmac::util::RWLock, public util::NonCopyable {
    DBC_FORCE_TEST(Context)
protected:
    /** Accelerator associated to the context */
    Accelerator &acc_;
    /** Execution mode owning the context */
    Mode &mode_;
    /** Context ID */
    unsigned id_;

    /**
     * Constructs a context for the calling thread on the given accelerator
     *
     * \param mode Reference to the parent mode of the context
     * \param id Context identifier
     */
    Context(Mode &mode, unsigned id);

    /**
     * Destroys the resources used by the context
     */
    virtual ~Context();

public:
    /**
     * Initialization method called on library initialization
     */
    static void init();

    /**
     * Copies size bytes from host memory to accelerator memory
     *
     * \param acc Destination pointer to accelerator memory
     * \param host Source pointer to host memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    TESTABLE VIRTUAL gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);
    /**
     * Copies size bytes from accelerator memory to host memory
     *
     * \param host Destination pointer to host memory
     * \param acc Source pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    TESTABLE VIRTUAL gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);
    /**
     * Copies size bytes from accelerator memory to accelerator memory
     *
     * \param dst Destination pointer to accelerator memory
     * \param src Source pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    TESTABLE VIRTUAL gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    /**
     * Fills size bytes of accelerator memory with the given value
     *
     * \param addr Pointer to accelerator memory
     * \param c Value to be used to fill the memory
     * \param size Number of bytes to be filled
     * \return Error code
     */
    virtual gmacError_t memset(accptr_t addr, int c, size_t size) = 0;

    /**
     * Waits for pending memory transfers before kernel execution
     *
     * \return Error code
     */
    virtual gmacError_t prepareForCall() = 0;

    /**
     * Waits for all kernels to finish execution on the accelerator
     *
     * \return Error code returned by the kernel
     */
    virtual gmacError_t waitForCall() = 0;

    /**
     * Waits for a kernel to finish execution on the accelerator
     * 
     * \param launch Kernel to wait for
     * \return Error code returned by the kernel
     */
    virtual gmacError_t waitForCall(core::hpe::KernelLaunch &launch) = 0;
};

}}}

#ifdef USE_DBC
#include "core/hpe/dbc/Context.h"
#endif

#endif
