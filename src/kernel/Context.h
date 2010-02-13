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

#include <debug.h>
#include <params.h>

#include <kernel/Process.h>
#include <kernel/Accelerator.h>

#include <gmac/gmac.h>
#include <memory/Map.h>
#include <memory/PageTable.h>


namespace gmac {

extern size_t paramBufferPageLockedSize;

namespace memory { class Manager; }

/*!
	\brief Generic Context Class
*/
class Context {
protected:
	/*!
		\brief Last error on context
	*/
	gmacError_t _error;

	/*!
		\brief Per-thread key to store context
	*/
	friend void contextInit(void);
	friend class memory::Manager;
	friend class Process;
	static PRIVATE(key);
	static PRIVATE(keyParent);
	static unsigned _next;
    static util::Lock lockCreate;

	unsigned _id;
	
	/*!
		\brief Memory map for the context
	*/
	memory::Map _mm;

	/*!
		\brief Returns a reference to the context memory map
	*/
	memory::Map &mm() { return _mm; }

	/*!
		\brief Returns a constant reference to the context memory map
	*/
	const memory::Map &mm() const { return _mm; }

	/*!
		\brief Accelerator where the context is attached
	*/
	Accelerator &acc;

    void * _bufferPageLocked;
    size_t _bufferPageLockedSize;

	void enable();

	Context(Accelerator &acc);

	virtual ~Context();


public:
    static Context *create(int acc = -1);

    /*! Gets the Context associated to the calling thread.
     *
     * If it does not have an associated Context, one new Context is created
     */
	static Context *current();

    /*! Checks whether the calling thread has an associated Context
     */
    static bool hasCurrent();

    /*! Initializes the per-thread private variables of the calling thread.
     *
     * This method must be called as soon as a new thread has been created
	 * \param parent Pointer to the parent thread. Used during Context cloning
    */
    static void initThread(Context *parent);

	void init();

	void destroy();
	
	/*!
		\brief Locks the context
	*/
	virtual void lock() = 0;
	
	/*!
		\brief Releases the context
	*/
	virtual void unlock() = 0;

	/*!
		\brief Allocates memory on the accelerator memory 
		\param addr Pointer to memory address to store the accelerator memory
		\param size Size, in bytes, to be allocated
	*/
	virtual gmacError_t malloc(void **addr, size_t size) = 0;

	/*!
		\brief Releases memory previously allocated by Malloc
		\param addr Starting memory address to be released
	*/
	virtual gmacError_t free(void *addr) = 0;

	/*!
		\brief Allocates page locked host memory and makes it accesible from the accelerator
		\param host   Pointer to memory address to store the memory
		\param device Pointer to memory address from the accelerator. If device is NULL, no mapping is performed
		\param size Size, in bytes, to be allocated
	*/
	virtual gmacError_t hostAlloc(void **host, void **device, size_t size) = 0;

	virtual gmacError_t hostMemAlign(void **host, void **device, size_t size) = 0;

	virtual gmacError_t hostMap(void *host, void **device, size_t size) = 0;

	/*!
		\bried Releases system memory accesible from the accelerator
		\param addr Starting memory address to be released
	*/
	virtual gmacError_t hostFree(void *addr) = 0;
	
	/*!
		\brief Copies data from system memory to accelerator memory
		\param dev Destination accelerator memory address
		\param host Source system memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyToDevice(void *dev, const void *host,
			size_t size) = 0;

	/*!
		\brief Copies data from accelerator memory to system memory
		\param host Destination system memory address
		\param dev Source accelerator memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyToHost(void *host, const void *dev,
			size_t size) = 0;

	/*!
		\brief Copies data from accelerator memory to accelerator memory
		\param src Source accelerator memory address
		\param dst Destination accelerator memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyDevice(void *dst, const void *src,
			size_t size) = 0;

	/*!
		\brief Copies data from system memory to accelerator memory
				asynchronously
		\param dev Destination accelerator memory address
		\param host Source system memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyToDeviceAsync(void *dev, const void *host,
			size_t size) = 0;


	/*!
		\brief Copies data from accelerator memory to system memory
				asynchronously
		\param host Destination host memory address
		\param dev Source host memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyToHostAsync(void *host, const void *dev,
			size_t size) = 0;

    /// \todo How-to enable features only supported in some CUDA SDK versions
    ///       like the following:
#if 0
    /*!
		\brief Copies data from accelerator memory to accelerator memory
		\param src Source accelerator memory address
		\param dst Destination accelerator memory address
		\param size Size, in bytes, to be copied
	*/
	virtual gmacError_t copyDeviceAsync(void *dst, const void *src,
			size_t size) = 0;
#endif


	/*!
		\brief Initializes accelerator memory
		\param addr Accelerator memory address
		\param value Value used to initialize memory
		\param size Size, in bytes, to be initialized
	*/
	virtual gmacError_t memset(void *dev, int value, size_t size) = 0;


	/*!
		\brief Launches the execution of a kernel
		\param kernel Kernel to be launched
	*/
	virtual gmacError_t launch(const char *kernel) = 0;

	/*!
		\brief Waits for kernel execution
	*/
	virtual gmacError_t sync() = 0;

	virtual bool async() const = 0;

	/*!
		\brief Waits for a memory transfer to host
	*/
	virtual gmacError_t syncToHost()   = 0;
	/*!
		\brief Waits for a memory transfer to device
	*/
	virtual gmacError_t syncToDevice() = 0;
#if 0
	/*!
		\brief Waits for kernel execution
	*/
	virtual gmacError_t syncDevice()   = 0;
#endif

	/*!
		\brief Returns last error
	*/
	gmacError_t error() const;

	virtual void flush() = 0;
	virtual void invalidate() = 0;

	unsigned id() const;
    /*!
		\brief Gets the page-locked buffer associated to the Context (if supported)
	*/
    void * bufferPageLocked() const;
    /*!
		\brief Gets the size in bytes of the page-locked buffer associated to the Context (or 0 if not supported)
	*/
    size_t bufferPageLockedSize() const;
};

#include "Context.ipp"

}


#endif
