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

#ifndef GMAC_MEMORY_DISTRIBUTEDBLOCK_H_
#define GMAC_MEMORY_DISTRIBUTEDBLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"

#include "Block.h"
#include "StateBlock.h"

namespace __impl { 

namespace core {
	class Mode;
}

namespace memory {

template<typename T>
class GMAC_LOCAL DistributedBlock : public StateBlock<T> {
protected:
	typedef std::map<core::Mode *, uint8_t *> AcceleratorMap;
	AcceleratorMap acceleratorAddr_;

public:
    //! Default construcutor
    /*!
        \param protocol Memory coherence protocol used by the block
        \param owner Initial ownwer of the memory block
        \param hostAddr Host memory address for applications to accesss the block
        \param shadowAddr Shadow host memory mapping that is always read/write
        \param acceleratorAddr Accelerator memory address for applications to accesss the block
        \param size Size (in bytes) of the memory block
        \param init Initial block state
    */
	DistributedBlock(Protocol &protocol, core::Mode &owner, uint8_t *shadowAddr,
		uint8_t *hostAddr, uint8_t *acceleratorAddr, size_t size, T init);

    //! Default destructor
    virtual ~DistributedBlock();

	void addOwner(core::Mode &owner, uint8_t *value);

	void removeOwner(core::Mode &owner);

	core::Mode &owner() const;

	void *acceleratorAddr(const void *addr) const;

	gmacError_t toHost() const;

	gmacError_t toAccelerator();

    gmacError_t copyToHost(const void *src, size_t size, 
        unsigned blockOffset = 0) const;

	gmacError_t copyToHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const;

    gmacError_t copyToAccelerator(const void *src, size_t size, 
        unsigned blockOffset = 0) const;

	gmacError_t copyToAccelerator(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const;
	
    gmacError_t copyFromHost(void *dst, size_t size,
        unsigned blockOffset = 0) const;

	gmacError_t copyFromHost(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const;

    gmacError_t copyFromAccelerator(void *dst, size_t size,
        unsigned blockOffset = 0) const;

	gmacError_t copyFromAccelerator(core::IOBuffer &buffer, size_t size, 
		unsigned bufferOffset = 0, unsigned blockOffset = 0) const;

    gmacError_t hostMemset(int v, size_t size, 
        unsigned blockOffset = 0) const;

    gmacError_t acceleratorMemset(int v, size_t size,
        unsigned blockOffset = 0) const;
};


}}

#include "DistributedBlock-impl.h"

#endif
