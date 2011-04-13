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

#include "Kernel.h"

namespace __impl {

namespace core {
class IOBuffer;
}

namespace opencl {

class IOBuffer;


namespace hpe {
class Accelerator;
class Mode;

class GMAC_LOCAL Context : public gmac::core::hpe::Context {
protected:
	static const unsigned USleepLaunch_ = 100;

	typedef std::map<void *, void *> AddressMap;
	static AddressMap HostMem_;

    cl_command_queue streamLaunch_;
    cl_command_queue streamToAccelerator_;
    cl_command_queue streamToHost_;
    cl_command_queue streamAccelerator_;

    IOBuffer *buffer_;

    void setupCLstreams();
    void cleanCLstreams();
    gmacError_t syncCLstream(cl_command_queue stream);

public:
	Context(Accelerator &acc, Mode &mode);
	~Context();

	gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);
	gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);
	gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    gmacError_t memset(accptr_t addr, int c, size_t size);

    core::hpe::KernelLaunch &launch(core::hpe::Kernel &kernel);
    gmacError_t prepareForCall();
    gmacError_t waitForCall();
    gmacError_t waitForCall(core::hpe::KernelLaunch &launch);

    gmacError_t bufferToAccelerator(accptr_t dst, core::IOBuffer &buffer, size_t size, size_t off = 0);
    gmacError_t acceleratorToBuffer(core::IOBuffer &buffer, const accptr_t dst, size_t size, size_t off = 0);
    gmacError_t waitAccelerator();

    const cl_command_queue eventStream() const;

    Accelerator & accelerator();
    gmacError_t waitForEvent(cl_event e);
};

}}}

#include "Context-impl.h"

#endif
