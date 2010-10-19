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

#ifndef GMAC_CORE_CONTEXT_H_
#define GMAC_CORE_CONTEXT_H_

#include "config/common.h"

#include "include/gmac-types.h"
#include "util/Logger.h"
#include "util/Private.h"

namespace gmac {

class Accelerator;
class Kernel;
class KernelLaunch;

/*!
	\brief Generic Context Class
*/
class GMAC_LOCAL Context : public util::RWLock, public util::Logger {
protected:
    Accelerator &acc_;
    unsigned id_;

	Context(Accelerator &acc, unsigned id);
public:
	virtual ~Context();

    static void init();

	virtual gmacError_t copyToAccelerator(void *dev, const void *host, size_t size);
	virtual gmacError_t copyToHost(void *host, const void *dev, size_t size);
	virtual gmacError_t copyAccelerator(void *dst, const void *src, size_t size);

    virtual gmacError_t memset(void *addr, int c, size_t size) = 0;

    virtual gmac::KernelLaunch &launch(gmac::Kernel &kernel) = 0;
    virtual gmacError_t sync() = 0;
};

}

#endif
