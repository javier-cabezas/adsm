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

#ifndef GMAC_API_OPENCL_ACCELERATOR_H_
#define GMAC_API_OPENCL_ACCELERATOR_H_

#include <CL/cl.h>

#include <list>
#include <map>
#include <utility>
#include <vector>

#include "config/common.h"
#include "core/Accelerator.h"
#include "Kernel.h"
#include "util/Lock.h"

namespace __impl { namespace opencl {
class IOBuffer;
class KernelLaunch;
class Mode;

class GMAC_LOCAL CommandList :
    protected std::list<cl_command_queue>,
    protected gmac::util::RWLock {
protected:
    typedef std::list<cl_command_queue> Parent;
public:
    CommandList() : RWLock("CommandList") {}
    virtual ~CommandList();

    void add(cl_command_queue stream);
    void remove(cl_command_queue stream);
    cl_command_queue &front();

    cl_int sync() const;
};

class GMAC_LOCAL HostMap :
    protected std::map<hostptr_t, std::pair<cl_mem, size_t> >,
    protected gmac::util::RWLock {
protected:
    typedef std::map<hostptr_t, std::pair<cl_mem, size_t> > Parent;
public:
    HostMap() : RWLock("HostMap") { }
    virtual ~HostMap();

    void insert(hostptr_t host, cl_mem acc, size_t size);
    void remove(hostptr_t host);

    bool translate(hostptr_t host, cl_mem &acc, size_t &size) const;
};

class GMAC_LOCAL Accelerator : public gmac::core::Accelerator {
protected:
    typedef std::map<Accelerator *, std::vector<cl_program> > AcceleratorMap;
    static AcceleratorMap *Accelerators_;

    cl_platform_id platform_;
    cl_device_id device_;

    cl_context ctx_;
    CommandList cmd_;
    static HostMap *GlobalHostMap_;
    HostMap localHostMap_;
    HostMap localHostAlloc_;

public:
    Accelerator(int n, cl_platform_id platform, cl_device_id device);
    ~Accelerator();

    cl_device_id device() const;

    static void init();

    static void addAccelerator(Accelerator &acc);
    Kernel *getKernel(gmacKernel_t k);

    static gmacError_t prepareEmbeddedCLCode();

    static gmacError_t prepareCLCode(const char *code, const char *flags);
    static gmacError_t prepareCLBinary(const unsigned char *binary, size_t size, const char *flags);

    gmac::core::Mode *createMode(core::Process &proc);

    gmacError_t malloc(accptr_t &addr, size_t size, unsigned align = 1);
    gmacError_t free(accptr_t addr);

    /* Synchronous interface */
    gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::Mode &mode);
    gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size, core::Mode &mode);
    gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    /* Asynchronous interface */
    gmacError_t copyToAcceleratorAsync(accptr_t acc, IOBuffer &buffer, size_t bufferOff, size_t count, core::Mode &mode, cl_command_queue stream);
    gmacError_t copyToHostAsync(IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, core::Mode &mode, cl_command_queue stream);

    cl_command_queue createCLstream();
    void destroyCLstream(cl_command_queue stream);
    cl_int queryCLstream(cl_command_queue stream);
    gmacError_t syncCLstream(cl_command_queue stream);
    cl_int queryCLevent(cl_event event);
    gmacError_t syncCLevent(cl_event event);
    gmacError_t timeCLevents(uint64_t &t, cl_event start, cl_event end);

    gmacError_t execute(KernelLaunch &launch);

    gmacError_t memset(accptr_t addr, int c, size_t size);

    gmacError_t sync();
    gmacError_t hostAlloc(hostptr_t &addr, size_t size);
    gmacError_t hostFree(hostptr_t addr);
    accptr_t hostMap(hostptr_t addr, size_t size);
    accptr_t hostMapAddr(hostptr_t addr);

    static gmacError_t error(cl_int r);

    void memInfo(size_t &free, size_t &total) const;
};

}}

#include "Accelerator-impl.h"

#ifdef USE_DBC
#include "dbc/Accelerator.h"
#endif


#endif
