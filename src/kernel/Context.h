/* Copyright (c) 2009 University of Illinois
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

#ifndef __KERNEL_CONTEXT_H_
#define __KERNEL_CONTEXT_H_

#include <threads.h>
#include <kernel/IAccelerator.h>
#include <memory/MemMap.h>

#define current static_cast<gmac::Context *>(PRIVATE_GET(gmac::Context::key))

namespace gmac {

/*!
	\brief Generic Context Class
*/
class Context : public IAccelerator {
protected:
	/*!
		\brief Last error on context
	*/
	gmacError_t _error;


	/*!
		\brief Memory map for the context
	*/
	MemMap _mm;

public:
	virtual ~Context() {};

	/*!
		\brief Per-thread key to store context
	*/
	static PRIVATE(key);

	/*!
		\brief Returns a reference to the context memory map
	*/
	MemMap &mm() { return _mm; }

	/*!
		\brief Returns a constant reference to the context memory map
	*/
	const MemMap &mm() const { return _mm; }

	/*!
		\brief Launches the execution of a kernel
		\param kernel Kernel to be launched
	*/
	virtual gmacError_t launch(const char *kernel) = 0;

	/*!
		\brief Records and error from a call
		\param e Error to record
	*/
	inline gmacError_t error(gmacError_t e) {
		_error = e;
		return e;
	}

	/*!
		\brief Returns last error
	*/
	inline gmacError_t error() const { return _error; }
};

};


#endif
