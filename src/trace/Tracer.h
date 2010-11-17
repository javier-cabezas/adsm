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

#ifndef GMAC_TRACE_TRACER_H_
#define GMAC_TRACE_TRACER_H_

#include "include/gmac/types.h"
#include "config/common.h"

#if defined(__GNUC__)
#define EnterCurrentFunction() EnterFunction(__func__)
#define ExitCurrentFunction()  ExitFunction(__func__)
#elif defined(_MSC_VER)
#define EnterCurrentFunction() EnterFunction(__FUNCTION__)
#define ExitCurrentFunction()  ExitFunction(__FUNCTION__)
#endif

namespace gmac { namespace trace {

typedef enum {
	Idle,
	Init,
	Running,
	Locked,
	Exclusive,
	IO
} State;
#if defined(USE_TRACE)
class GMAC_LOCAL Tracer {
protected:
	uint64_t timeMark() const;
	uint64_t base_;
public:
	Tracer();
	virtual void startThread(THREAD_T tid, const char *name) = 0;
	virtual void endThread(THREAD_T tid) = 0;

	virtual void enterFunction(THREAD_T tid, const char *name) = 0;
	virtual void exitFunction(THREAD_T tid, const char *name) = 0;

	virtual void setThreadState(THREAD_T tid, const State state) = 0;
};
#endif
void InitTracer();
void FiniTracer();

void StartThread(THREAD_T tid, const char *name);
void StartThread(const char *name);

void EndThread(THREAD_T tid);
void EndThread();

void EnterFunction(THREAD_T tid, const char *name);
void EnterFunction(const char *name);

void ExitFunction(THREAD_T tid, const char *name);
void ExitFunction(const char *name);

void SetThreadState(THREAD_T tid, const State &state);
void SetThreadState(const State &state);


}}

#include "Tracer-impl.h"

#endif
