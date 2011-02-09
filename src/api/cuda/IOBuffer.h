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

#ifndef GMAC_API_CUDA_IOBUFFER_H_
#define GMAC_API_CUDA_IOBUFFER_H_

#include <cuda.h>

#include "core/IOBuffer.h"
#include "config/common.h"

//include "core/dbc/IOBuffer.h"
//using  __impl::core::IOBuffer;
//using gmac::core::IOBuffer;

namespace __impl { namespace cuda {

class Mode;

class GMAC_LOCAL IOBuffer : public gmac::core::IOBuffer {
private:
    gmacError_t wait(bool fromCUDA);

protected:
    CUevent start_;
    CUevent end_;
    CUstream stream_;
    Mode *mode_;

    typedef std::map<Mode *, std::pair<CUevent, CUevent> > EventMap;
    EventMap map_;

public:
    IOBuffer(void *addr, size_t size) :
        gmac::core::IOBuffer(addr, size), mode_(NULL)
    {
    }

    ~IOBuffer()
    {
        EventMap::iterator it;
        for (it = map_.begin(); it != map_.end(); it++) {
            cuEventDestroy(it->second.first);
            cuEventDestroy(it->second.second);
        }
    }

    void toHost(Mode &mode, CUstream s);
    void toAccelerator(Mode &mode, CUstream s);

    void started();

    gmacError_t wait();
    gmacError_t waitFromCUDA();
};

}}

#include "IOBuffer-impl.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
