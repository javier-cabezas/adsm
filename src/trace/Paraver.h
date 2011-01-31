/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_TRACE_PARAVER_H_
#define GMAC_TRACE_PARAVER_H_

#if defined(USE_TRACE_PARAVER)
#include "Tracer.h"
#include "config/common.h"

#include "paraver/Trace.h"
#include "paraver/Names.h"
#include "paraver/Lock.h"

#include <iostream>

namespace __impl { namespace trace {

class GMAC_LOCAL Paraver : public Tracer {
protected:
    std::string baseName_, fileName_;
    paraver::TraceWriter trace_;

    typedef std::map<std::string, int32_t > FunctionMap;
    paraver::Lock mutex_;
    FunctionMap functions_;

    paraver::EventName *FunctionEvent_;
    typedef std::map<const State, paraver::StateName *> StateMap;
    StateMap states_;
    
public:
    Paraver();
    ~Paraver();

    void startThread(THREAD_T tid, const char *name);
    void endThread(THREAD_T tid);

    void enterFunction(THREAD_T tid, const char *name);
    void exitFunction(THREAD_T tid, const char *name);

    void setThreadState(THREAD_T tid, const State state);
    
    void dataCommunication(THREAD_T src, THREAD_T dst, uint64_t delta, size_t size);
};

} }

#endif

#endif
