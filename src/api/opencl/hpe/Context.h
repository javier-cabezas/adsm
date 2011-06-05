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

#ifndef GMAC_API_OPENCL_HPE_CONTEXT_H_
#define GMAC_API_OPENCL_HPE_CONTEXT_H_

#include <CL/cl.h>

#include <map>
#include <vector>

#include "config/common.h"
#include "config/config.h"

#include "core/hpe/Context.h"
#include "util/Lock.h"

namespace __impl {

namespace core {
class IOBuffer;
}

namespace opencl {

class IOBuffer;


namespace hpe {

class Accelerator;
class Mode;
class Kernel;
class KernelLaunch;


class GMAC_LOCAL Context : public gmac::core::hpe::Context {
    friend class ContextFactory;
protected:
    /** Delay for spin-locking */
	static const unsigned USleepLaunch_ = 100;

    /** OpenCL command queue to request operations */
    cl_command_queue stream_;

    /** I/O buffer used by the context for data transfers */
    IOBuffer *buffer_;

    /**
     * Wait for all commands in the OpenCL command queue to finish
     * \param stream OpenCL command queue
     * \return Error code
     */
    gmacError_t syncCLstream(cl_command_queue stream);

    /**
     * Default OpenCL context constructor
     * \param mode OpenCL execution mode associated to the context
     */
    Context(Mode &mode, cl_command_queue stream);

    /**
     * Default OpenCL context destructor
     */
    ~Context();

public:
    /**
     * Get the accelerator associated to the context
     * \return Reference to an OpenCL accelerator
     */
    Accelerator & accelerator();

    /**
     * Copy data from an I/O buffer to the accelerator memory
     * \param dst Accelerator memory address to copy data to
     * \param buffer I/O buffer to copy data from
     * \param size Size (in bytes) of the data to be transferred
     * \param off  Offset within the I/O buffer to start transferring data from
     */
    gmacError_t bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t size, size_t off = 0);

    /**
     * Copy data from the accelerator to an I/O buffer
     * \param buffer I/O buffer to copy the data to
     * \param dst Accelerator memory address to copy data from
     * \param size Size (in bytes) of the data to be copied
     * \param off Offset within the I/O buffer to start transferring data to
     */
    gmacError_t acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t dst, size_t size, size_t off = 0);


    /**
     * Create a descriptor of a kernel invocation
     * \param kernel OpenCL kernel to be executed
     * \return Descriptor of the kernel invocation
     */
    KernelLaunch &launch(Kernel &kernel);

    /**
     * Wait for the accelerator to finish activities in all OpenCL command queues
     * \return Error code
     */
    gmacError_t waitAccelerator();

    /**
     * Wait for an OpenCL event to be completed
     * \param e OpenCL event to wait for
     * \return Error code
     */
    gmacError_t waitForEvent(cl_event e);

    /**
     * Get the default OpenCL command queue to request events
     * \return Default OpenCL command queue
     */
    const cl_command_queue eventStream() const;


    /* core/hpe/Context.h Interface */
	gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);

	gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);

	gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    gmacError_t memset(accptr_t addr, int c, size_t size);

    gmacError_t prepareForCall();

    gmacError_t waitForCall();

    gmacError_t waitForCall(core::hpe::KernelLaunch &launch);
};

}}}

#include "Context-impl.h"

#endif
