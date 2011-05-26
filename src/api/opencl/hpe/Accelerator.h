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
WITH THE SOFTWARE.
*/

#ifndef GMAC_API_OPENCL_HPE_ACCELERATOR_H_
#define GMAC_API_OPENCL_HPE_ACCELERATOR_H_


#include <CL/cl.h>

#include <list>
#include <map>
#include <utility>
#include <vector>

#include "config/common.h"
#include "core/hpe/Accelerator.h"
#include "api/opencl/hpe/ModeFactory.h"
#include "util/Lock.h"

namespace __impl { namespace opencl {

class IOBuffer;

namespace hpe {

class Mode;

class KernelLaunch;
class Kernel;

/** A list of command queues */
class GMAC_LOCAL CommandList :
    protected std::list<cl_command_queue>,
    protected gmac::util::RWLock {
protected:
    /** Base type from STL */
    typedef std::list<cl_command_queue> Parent;
public:
    /** Default constructor */
    CommandList() : RWLock("CommandList") {}
    /** Default destructor */
    virtual ~CommandList();

    /** Add a command queue to the list
     * \param stream Command queue to be inserted
     */
    void add(cl_command_queue stream);

    /** Remove a command queue from the list
     * \param stream Command queue to be removed
     */
    void remove(cl_command_queue stream);

    /** Get the command queue from the front of the list
     * \return Command queue at the fron of the list
     */
    cl_command_queue &front();

    /** Wait for all command queue in the list to finish execution
     * \return Error code
     */
    cl_int sync() const;
};

/** A map of host memory addresses associated to OpenCL memory objects */
class GMAC_LOCAL HostMap :
    protected std::map<hostptr_t, std::pair<cl_mem, size_t> >,
    protected gmac::util::RWLock {
protected:
    /** Base type from STL */
    typedef std::map<hostptr_t, std::pair<cl_mem, size_t> > Parent;
public:
    /** Default constructor */
    HostMap() : RWLock("HostMap") { }
    /** Default destructor */
    virtual ~HostMap();

    /** Insert a new entry in the map
     * \param host Host memory address
     * \param acc OpenCL object
     * \param size (in bytes) of the object
     */
    void insert(hostptr_t host, cl_mem acc, size_t size);

    /** Remove an entry from the map
     * \param host Host memory address of the entry
     */
    void remove(hostptr_t host);

    /** Get the OpenCL memory object associated to a host memory address
     * \param host Host memory address
     * \param acc Reference to store the associated OpenCL memory object
     * \param size Reference to the size (in bytes) of the OpenCL memory object
     * \return True if the translation succeeded
     */
    bool translate(hostptr_t host, cl_mem &acc, size_t &size) const;
};

class GMAC_LOCAL CLMem :
    protected std::map<size_t, std::list<cl_mem> >,
    protected gmac::util::Lock {

    typedef std::list<cl_mem> CLMemList;
    typedef std::map<size_t, CLMemList> CLMemMap;
public:
    CLMem() : CLMemMap(), gmac::util::Lock("CLMem") {}

    ~CLMem()
    {
        CLMemMap::iterator it;
        
        for(it = begin(); it != end(); it++) {
            CLMemList::iterator it2;
            CLMemList list = it->second;
            for (it2 = list.begin(); it2 != list.end(); it2++) {
                cl_int ret;
                ret = clReleaseMemObject(*it2);
                ASSERTION(ret == CL_SUCCESS);
            }
        }
    }

    bool getCLMem(size_t size, cl_mem &mem)
    {
        lock(); 

        bool ret = false;

        CLMemMap::iterator it = find(size);
        
        if (it != end())  {
            CLMemList &list = it->second;
            if (it->second.size() > 0) {
                mem = list.front();
                list.pop_front();
                ret = true;
            }
        }
        unlock();

        return ret;
    }

    void putCLMem(size_t size, cl_mem mem)
    {
        lock(); 

        CLMemMap::iterator it = find(size);
        
        if (it != end())  {
            CLMemList &list = it->second;
            if (it->second.size() > 0) {
                list.push_back(mem);
            }
        } else {
            CLMemList list;
            list.push_back(mem);
            insert(CLMemMap::value_type(size, list));
        }
        unlock();
    }
};

/** An OpenCL capable accelerator */
class GMAC_LOCAL Accelerator :
    protected ModeFactory,
    public gmac::core::hpe::Accelerator {
protected:
    typedef std::map<Accelerator *, std::vector<cl_program> > AcceleratorMap;
    /** Map of the OpenCL accelerators in the system and the associated OpenCL programs */
    static AcceleratorMap *Accelerators_;
    /** Host memory allocations associated to any OpenCL accelerator */
    static HostMap *GlobalHostAlloc_;

    CLMem clMem_;

    /** OpenCL plaform ID for the accelertor */
    cl_platform_id platform_;
    /** OpenCL device ID for the accelerator */
    cl_device_id device_;

    /** OpenCL context associated to the accelerator */
    cl_context ctx_;
    /** List of command queues associated to the accelerator */
    CommandList cmd_;
    /** Host memory allocations associated to the accelerator */
    HostMap localHostAlloc_;

public:
    /** Default constructor
     * \param n Accelerator number
     * \param platform OpenCL platform ID for the accelerator
     * \param device OpenCL device ID for the accelerator
     */
    Accelerator(int n, cl_platform_id platform, cl_device_id device);
    /** Default destructor */
    virtual ~Accelerator();

    /**
     * Get the OpenCL device ID associated to the accelerator
     * \return OpenCL device ID
     */
    cl_device_id device() const;

    /**
     * Initialize the accelerator global data structures
     */
    static void init();

    /**
     * Get a GMAC error associated to an OpenCL error code
     * \param r OpenCL error code
     * \return GMAC error code
     */
    static gmacError_t error(cl_int r);

    /**
     * Add a new accelerator
     * \param acc Accelerator to be added
     */
    static void addAccelerator(Accelerator &acc);

    /**
     * Check for OpenCL code embedded in the binary
     * \return Error code
     */
    static gmacError_t prepareEmbeddedCLCode();

    /**
     * Make source OpenCL kernel available to all accelerators
     * \param code OpenCL source code
     * \param flags Compilation flags
     * \return Error code
     */
    static gmacError_t prepareCLCode(const char *code, const char *flags);

    /**
     * Make binary OpenCL kernel available to all accelerators
     * \param binary OpenCL binary code
     * \param size Size (in bytes) of the OpenCL binary code
     * \param flags Compilation flags
     * \return Error Code
     */
    static gmacError_t prepareCLBinary(const unsigned char *binary, size_t size, const char *flags);

    /**
     * Get a kernel from its ID
     * \param k Kernel ID
     * \return Kernel object
     */
    Kernel *getKernel(gmac_kernel_id_t k);

    /**
     *  Create a new execution mode for this accelerator
     * \param proc Process where to bind the execution mode
     * \return Execution mode
     */
    core::hpe::Mode *createMode(core::hpe::Process &proc);

    /**
     *  Get the OpenCL context associated to the accelerator
     * \return OpenCL context
     */
    const cl_context getCLcontext() const;

    /**
     * Allocate pinned accelerator-accessible host memory
     * \param addr Reference to store the memory address of the allocated memory
     * \param size Size (in bytes) of the memoty to be allocated
     * \return Error code
     */
    gmacError_t hostAlloc(hostptr_t &addr, size_t size);

    gmacError_t allocCLBuffer(cl_mem &mem, hostptr_t &addr, size_t size);

    /**
     * Release pinned accelerator-accessible host memory
     * \param addr Host memory address to be released
     * \return Error code
     */
    gmacError_t hostFree(hostptr_t addr);

    gmacError_t freeCLBuffer(cl_mem mem, size_t size);

    /**
     * Get the accelerator memory address where pinned host memory can be accessed
     * \param addr Host memory address to be mapped to the accelerator
     * \return Accelerator memory address
     */
    accptr_t hostMapAddr(hostptr_t addr);

    /**
     * Asynchronously copy an I/O buffer to the accelerator
     * \param acc Accelerator memory address where to copy the data to
     * \param buffer I/O buffer containing the data to be copied
     * \param bufferOff Offset from the starting of the I/O buffer to start copying data from
     * \param count Size (in bytes) to be copied
     * \param mode Execution mode associated to the data transfer
     * \param stream OpenCL command queue where to issue the data transfer request
     * \return Error code
     */
    gmacError_t copyToAcceleratorAsync(accptr_t acc, IOBuffer &buffer, size_t bufferOff, size_t count, core::hpe::Mode &mode, cl_command_queue stream);

    /**
     * Asynchronously copy data from accelerator to an I/O buffer
     * \param buffer I/O buffer where to copy the data to
     * \param bufferOff Offset from the starting of the I/O buffer to start copying data to
     * \param acc Accelerator memory address where to start copying data from
     * \param count Size (in bytes) to be copied
     * \param mode Execution mode associated to the data transfer
     * \param stream OpenCL command queue where to issue the data transfer request
     * \return Error code
     */
    gmacError_t copyToHostAsync(IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, core::hpe::Mode &mode, cl_command_queue stream);

    /**
     * Create an OpenCL command queue
     * \return OpenCL command queue
     */
    cl_command_queue createCLstream();

    /**
     * Destroy an OpenCL command queue
     * \param stream OpenCL command queue to be destroyed
     */
    void destroyCLstream(cl_command_queue stream);

    /**
     * Query for the state of an OpenCL command queue
     * \param stream OpenCL command queue to query for its state
     * \return OpenCL command queue state
     */
    cl_int queryCLstream(cl_command_queue stream);

    /**
     * Wait for all commands in a command queue to be completed
     * \param stream OpenCL command queue
     * \return Error code
     */
    gmacError_t syncCLstream(cl_command_queue stream);

    /**
     * Query for the state of an OpenCL event
     * \param event OpenCL event
     * \return OpenCL event status
     */
    cl_int queryCLevent(cl_event event);

    /**
     * Wait for an OpenCL event to be completed
     * \param event OpenCL event
     * \return Error code
     */
    gmacError_t syncCLevent(cl_event event);

    /**
     * Calculate the time elapsed between to events happend
     * \param t Reference to store the elapsed time
     * \param start Event when the elapsed time start
     * \param end Event when the elapsed time end
     * \return Error code
     */
    gmacError_t timeCLevents(uint64_t &t, cl_event start, cl_event end);

    /**
     * Execute a kernel
     * \param launch Descriptor of the kernel to be executed
     * \return Error code
     */
    gmacError_t execute(KernelLaunch &launch);


    /* core/hpe/Accelerator.h Interface */
    gmacError_t map(accptr_t &dst, hostptr_t src, size_t size, unsigned align = 1);

    gmacError_t unmap(hostptr_t addr, size_t size);

    gmacError_t sync();

    gmacError_t
        copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode);

    gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::hpe::Mode &mode);

    gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);
    gmacError_t memset(accptr_t addr, int c, size_t size);
    void memInfo(size_t &free, size_t &total) const;

};

}}}

#include "Accelerator-impl.h"

#ifdef USE_DBC
#include "dbc/Accelerator.h"
#endif


#endif
