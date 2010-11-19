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

#ifndef GMAC_MEMORY_PROTOCOL_LAZY_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_H_

#include "config/common.h"
#include "include/gmac/types.h"

#include "memory/Handler.h"
#include "memory/Protocol.h"
#include "util/Lock.h"

#include "BlockListMap.h"

namespace __impl {

namespace core {
    class IOBuffer;
    class Mode;
}
    
namespace memory {
class Object;
class Block;
template<typename T> class StateBlock;

namespace protocol { 

class GMAC_LOCAL Lazy : public Protocol, Handler, gmac::util::RWLock {
public:
    typedef enum {
        Invalid,
        ReadOnly,
        Dirty,
        HostOnly
    } State;
protected:
	State state(GmacProtection prot) const;

    unsigned limit_;
    BlockListMap dbl_;

	gmacError_t release(StateBlock<State> &block);
    void addDirty(Block &block);
public:
    Lazy(unsigned limit);
    virtual ~Lazy();

    // Protocol Interface
    memory::Object *createObject(size_t size, void *cpuPtr, GmacProtection prot);
	memory::Object *createGlobalObject(size_t size, void *cpuPtr, GmacProtection prot);
	void deleteObject(Object &obj);

    gmacError_t signalRead(Block &block);
    gmacError_t signalWrite(Block &block);

    gmacError_t acquire(Block &obj);
#ifdef USE_VM
    gmacError_t acquireWithBitmap(const Object &obj);
#endif
    gmacError_t release();
    gmacError_t remove(Block &block);

	gmacError_t toHost(Block &block);
    gmacError_t toDevice(Block &block);

	gmacError_t copyToBuffer(const Block &block, core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset, unsigned objectOffset) const;
	
	gmacError_t copyFromBuffer(const Block &block, core::IOBuffer &buffer, size_t size,
		unsigned bufferOffset, unsigned objectOffset) const;

};

}}}

#ifdef USE_DBC
//#include "dbc/Lazy.h"
#endif

#endif
