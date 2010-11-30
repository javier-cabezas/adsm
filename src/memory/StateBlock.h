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

#ifndef GMAC_MEMORY_STATEBLOCK_H_
#define GMAC_MEMORY_STATEBLOCK_H_

#include "config/common.h"
#include "config/config.h"

#include "include/gmac/types.h"

namespace __impl { namespace memory {

template<typename T>
class GMAC_LOCAL StateBlock : public Block {
protected:
    //! Block state
	T state_;    

    //! Default construcutor
    /*!
        \param protocol Memory coherence protocol used by the block
        \param addr Host memory address for applications to accesss the block
        \param shadow Shadow host memory mapping that is always read/write
        \param size Size (in bytes) of the memory block
        \param init Initial block state
    */
	StateBlock(Protocol &protocol, uint8_t *addr, uint8_t *shadow, size_t size, T init);
public:
    //! Get block state
    /*!
        \return Block state
    */
	const T &state() const;

    //! Set block state
    /*!
        \param s New block state
    */
	void state(const T &s);
};

}}

#include "StateBlock-impl.h"


#endif
