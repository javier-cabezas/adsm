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

#ifndef GMAC_CORE_MODE_H_
#define GMAC_CORE_MODE_H_

#include "config/common.h"

#include "core/Accelerator.h"
#include "core/Context.h"
#include "core/allocator/Buddy.h"

#ifdef USE_VM
#include "memory/Bitmap.h"
#endif

#include "memory/Manager.h"
#include "memory/Map.h"

#include "util/Lock.h"
#include "util/NonCopyable.h"
#include "util/Reference.h"
#include "util/Private.h"

namespace __impl {

namespace memory { class Map; class Object; class Block; }

namespace core {

class Context;
class IOBuffer;
class KernelLaunch;
class Process;

class GMAC_LOCAL ContextMap : protected std::map<THREAD_T, Context *>, gmac::util::RWLock {
protected:
    typedef std::map<THREAD_T, Context *> Parent;
public:
    ContextMap() : gmac::util::RWLock("ContextMap") {}
    void add(THREAD_T id, Context *ctx);
    Context *find(THREAD_T id);
    void remove(THREAD_T id);
    void clean();

    gmacError_t sync();
};

/**
 * A Mode represents the address space of a thread in an accelerator. Each
 * thread has one mode per accelerator type in the system
 */
class GMAC_LOCAL Mode : public util::Reference, public util::NonCopyable {
protected:
    static util::Private<Mode> key;
    static unsigned next;

    unsigned id_;

    Process &proc_;
    // Must be a pointer since the Mode can change the accelerator on which it is running
    Accelerator *acc_;

    memory::Map map_;
    memory::Protocol *protocol_;

    allocator::Buddy *ioMemory_;

    bool releasedObjects_;

#ifdef USE_SUBBLOCK_TRACKING
    __impl::memory::vm::BitmapHost hostBitmap_;
#else
#ifdef USE_VM
    __impl::memory::vm::BitmapShared acceleratorBitmap_;
#endif
#endif

    ContextMap contextMap_;

    typedef std::map<gmacKernel_t, Kernel *> KernelMap;
    KernelMap kernels_;

    virtual void switchIn() = 0;
    virtual void switchOut() = 0;

    virtual void reload() = 0;
    virtual Context &getContext() = 0;

    gmacError_t error_;

    void cleanUpContexts();
    gmacError_t cleanUp();
public:
    /**
     * Mode constructor
     *
     * \param proc Reference to the process which the mode belongs to
     * \param acc Reference to the accelerator in which the mode will perform
     *            the allocations
    */
    Mode(Process &proc, Accelerator &acc);

    /**
     * Mode destructor
     */
    virtual ~Mode();

    /**
     * Function called on process initialization to register thread-specific
     * variables
     */
    static void init();

    /**
     * Function called on thread creation to initialize thread-specific
     * variables
     */
    static void initThread();

    /**
     * Function called on thread destruction to release the Mode resources
     */
    static void finiThread();

    /**
     * Gets the Mode for the calling thread
     * \return A reference to the Mode for the calling thread. If the thread has
     *         no Mode yet, a new one is created
     */
    static Mode &current();

    /**
     * Tells if the calling thread already has a Mode assigned
     * \return A boolean that tells if the calling thread already has a Mode
     *         assigned
     */
    static bool hasCurrent();

    /**
     * Gets a reference to the memory protocol used by the mode
     * \return A reference to the memory protocol used by the mode
     */
    memory::Protocol &protocol();

    /**
     * Gets a numeric identifier for the mode. This identifier must be unique.
     * \return A numeric identifier for the mode
     */
    unsigned id() const;

    /**
     * Gets a reference to the accelerator which the mode belongs to
     * \return A reference to the accelerator which the mode belongs to
     */
    Accelerator &getAccelerator() const;

    /**
     * Attaches the execution mode to the current thread
     */
    void attach();

    /**
     * Dettaches the execution mode to the current thread
     */
    void detach();

    /**
     * Adds an object to the map of the mode
     * \param obj A reference to the object to be added
     */
    void addObject(memory::Object &obj);

    /**
     * Removes an object from the map of the mode
     * \param obj A reference to the object to be removed
     */
    void removeObject(memory::Object &obj);

    /**
     * Gets the first object that belongs to the memory range
     * \param addr Starting address of the memory range
     * \param size Size of the memory range
     * \return A pointer of the Object that contains the address or NULL if
     * there is no Object at that address
     */
    memory::Object *getObject(const hostptr_t addr, size_t size = 0) const;

    /**
     * Applies a constant memory operation to all the objects that belong to
     * the mode
     * \param op Memory operation to be executed
     *   \sa __impl::memory::Object::acquire
     *   \sa __impl::memory::Object::toHost
     *   \sa __impl::memory::Object::toAccelerator
     * \return Error code
     */
    gmacError_t forEachObject(memory::ObjectMap::ConstObjectOp op) const;

    /**
     * Allocates memory on the accelerator memory
     * \param addr Reference to a pointer where to store the accelerator
     * address of the allocation
     * \param size Size of the allocation
     * \param align Alignment of the memory allocation. This value must be a
     * power of two
     * \return Error code
     */
    gmacError_t malloc(accptr_t &addr, size_t size, unsigned align = 1);

    /**
     * Releases memory previously allocated by malloc
     * \param addr Accelerator memory allocation to be freed
     * \return Error code
     */
    gmacError_t free(accptr_t addr);

    /**
     * Copies data from system memory to accelerator memory
     * \param acc Destination accelerator pointer
     * \param host Source host pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);

    /**
     * Copies data from accelerator memory to system memory
     * \param host Destination host pointer
     * \param acc Source accelerator pointer
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);

    /** Copies data from accelerator memory to accelerator memory
     * \param dst Destination accelerator memory
     * \param src Source accelerator memory
     * \param size Number of bytes to be copied
     * \return Error code
     */
    gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    /**
     * Sets the contents of accelerator memory
     * \param addr Pointer to the accelerator memory to be set
     * \param c Value used to fill the memory
     * \param size Number of bytes to be set
     * \return Error code
     */
    gmacError_t memset(accptr_t addr, int c, size_t size);

    /**
     * Creates a KernelLaunch object that can be executed by the mode
     * \param kernel Handler of the kernel to be launched
     * \return Reference to the KernelLaunch object
     */
    virtual KernelLaunch &launch(gmacKernel_t kernel) = 0;

    /**
     * Executes a kernel using a KernelLaunch object
     * \param launch Reference to a KernelLaunch object
     * \return Error code
     */
    virtual gmacError_t execute(KernelLaunch &launch) = 0;

    /**
     * Waits for kernel execution
     * \return Error code
     */
    gmacError_t sync();

    /**
     * Creates an IOBuffer
     * \param size Minimum size of the buffer
     * \return A pointer to the created IOBuffer or NULL if there is not enough
     *         memory
     */
    virtual IOBuffer *createIOBuffer(size_t size) = 0;

    /**
     * Destroys an IOBuffer
     * \param buffer Pointer to the buffer to be destroyed
     */
    virtual void destroyIOBuffer(IOBuffer *buffer) = 0;

    /** Copies size bytes from an IOBuffer to accelerator memory
     * \param dst Pointer to accelerator memory
     * \param buffer Reference to the source IOBuffer
     * \param size Number of bytes to be copied
     * \param off Offset within the buffer
     */
    virtual gmacError_t bufferToAccelerator(accptr_t dst, IOBuffer &buffer, size_t size, size_t off = 0) = 0;

    /**
     * Copies size bytes from accelerator memory to a IOBuffer
     * \param buffer Reference to the destination buffer
     * \param dst Pointer to accelerator memory
     * \param size Number of bytes to be copied
     * \param off Offset within the buffer
     */
    virtual gmacError_t acceleratorToBuffer(IOBuffer &buffer, const accptr_t dst, size_t size, size_t off = 0) = 0;

    void kernel(gmacKernel_t k, Kernel &kernel);
    //Kernel * kernel(gmacKernel_t k);

    /**
     * Returns the last error code
     * \return The last error code
     */
    gmacError_t error() const;

    /**
     * Sets up the last error code
     * \param err Error code
     */
    void error(gmacError_t err);

    /**
     * Moves the mode to accelerator acc
     * \param acc Accelerator to move the mode to
     * \return Error code
     */
    gmacError_t moveTo(Accelerator &acc);

    /**
     * Tells if the objects of the mode have been already released to the
     * accelerator
     * \return Boolean that tells if objects of the mode have been already
     * released to the accelerator
     */
    bool releasedObjects() const;

    /**
     * Releases the ownership of the objects of the mode to the accelerator
     */
    void releaseObjects();

    /**
     * Acquires the ownership of the objects of the mode from the accelerator
     */
    void acquireObjects();

    /**
     * Returns the process which the mode belongs to
     * \return A reference to the process which the mode belongs to
     */
    Process &process();

    /** Returns the process which the mode belongs to
     * \return A constant reference to the process which the mode belongs to
     */
    const Process &process() const;

#ifdef USE_SUBBLOCK_TRACKING
    memory::vm::BitmapHost &hostDirtyBitmap();
    const memory::vm::BitmapHost &hostDirtyBitmap() const;
#else
#ifdef USE_VM
    memory::vm::BitmapShared &acceleratorDirtyBitmap();
    const memory::vm::BitmapShared &acceleratorDirtyBitmap() const;
#endif
#endif

};

}}

#include "Mode-impl.h"

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
