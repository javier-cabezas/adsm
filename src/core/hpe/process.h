/* Copyright (c) 2009, 2010, 2011 University of Illinois
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

#ifndef GMAC_CORE_HPE_PROCESS_H_
#define GMAC_CORE_HPE_PROCESS_H_

#include <list>
#include <map>
#include <vector>

#include "config/common.h"
#include "config/order.h"
#include "include/gmac/types.h"

#include "util/lock.h"
#include "util/Private.h"
#include "util/UniquePtr.h"

#include "core/process.h"

#include "resource_manager.h"

namespace __impl { namespace core {

class vdevice;

namespace hpe {

class thread;
class vdevice;

/** Map that contains in which accelerator resides a mode */
class GMAC_LOCAL map_vdevice : private std::map<vdevice *, unsigned>,
                              private gmac::util::mutex<map_vdevice> {
    friend class process;
private:
    typedef std::map<vdevice *, unsigned> Parent;
    typedef gmac::util::mutex<map_vdevice> Lock;

public:
    /** Constructs the vdeviceMap */
    map_vdevice();

    typedef Parent::iterator iterator;
    typedef Parent::const_iterator const_iterator;

    /**
     * Inserts a mode/accelerator pair in the map
     * \param mode vdevice to be inserted
     * \return A pair that contains the position where the items have been
     * allocated and a boolean that tells if the items have been actually
     * inserted
     */
    std::pair<iterator, bool> insert(vdevice *mode);
    void remove(vdevice &mode);
};

/** Represents the resources used by a running process */
class GMAC_LOCAL process : public core::process,
                           public gmac::util::lock_rw<process> {
    DBC_FORCE_TEST(process)
protected:
	typedef gmac::util::lock_rw<process> Lock;

    unsigned current_;

    resource_manager resourceManager_;

    typedef std::map<THREAD_T, thread *> map_thread;
    map_thread mapThreads_; 

    /**
     * Destroys the process and releases the resources used by it
     */
    virtual ~process();

public:
    /**
     * Constructs the process
     */
    process();

    void init();

    /**
     * Registers a new thread in the process
     */
    TESTABLE void initThread(bool userThread, THREAD_T tid);

    /**
     * Unregisters a thread from the process
     */
    TESTABLE void finiThread(bool userThread);

    /**
     * Waits for pending operations before a kernel call (needed for distributed objects)
     *
     * \return gmacSuccess on success, an error code otherwise
     */
    gmacError_t prepareForCall();

    resource_manager &get_resource_manager();
};

}}}

#include "process-impl.h"

#ifdef USE_DBC
#include "core/hpe/dbc/process.h"
#endif

#endif
