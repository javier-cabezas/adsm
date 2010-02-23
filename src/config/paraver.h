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

#ifndef __CONFIG_PARAVER_H
#define __CONFIG_PARAVER_H

namespace paraver {
enum FunctionName {
    FuncAccMalloc        = 1,  // 1
    FuncAccFree          = 2,  // 2
    FuncAccHostDevice    = 3,  // 3
    FuncAccDeviceHost    = 4,  // 4
    FuncAccDeviceDevice  = 5,  // 5
    FuncAccLaunch        = 6,  // 6
    FuncAccSync          = 7,  // 7
    FuncGmacMalloc       = 8,  // 8
    FuncGmacGlobalMalloc = 9,  // 9
    FuncGmacFree         = 10, // 10
    FuncGmacLaunch       = 11, // 11
    FuncGmacSync         = 12, // 12
    FuncGmacSignal       = 13, // 13
    FuncGmacAccs         = 14, // 14
    FuncGmacSetAffinity  = 15, // 15
    FuncGmacClear        = 16, // 16
    FuncGmacBind         = 17, // 17
    FuncGmacUnbind       = 18, // 18
    FuncVmAlloc          = 19, // 19
    FuncVmFree           = 20, // 20
    FuncVmFlush          = 21, // 21
    FuncVmSync           = 22  // 22
};

enum LockName {
    LockMmLocal      = 1,  // 1
    LockMmGlobal     = 2,  // 2
    LockMmShared     = 3,  // 3
    LockPageTable    = 4,  // 4
    LockCtxLocal     = 5,  // 5
    LockCtxGlobal    = 6,  // 6
    LockCtxCreate    = 7,  // 7
    LockQueue        = 8,  // 8
    LockIoHost       = 9,  // 9
    LockIoDevice     = 10, // 10
    LockProcess      = 11, // 11
    LockWriteMutex   = 12, // 12
    LockRollingMap   = 13, // 13
    LockRollingBuffer= 14, // 14
    LockManager      = 15, // 15
    LockThreadQueue  = 16, // 16
    LockRegion       = 17, // 17
    LockPthread      = 18, // 18
    LockShMap        = 19, // 19
    LockContextList  = 20, // 20
    LockBlockList    = 21, // 21
    LockQueueMap     = 22  // 22
};
}

#ifdef PARAVER
#include <paraver/Trace.h>
#include <paraver/Names.h>

namespace paraver {

extern Trace *trace;

EVENT(Function);
EVENT(HostDeviceCopy);
EVENT(DeviceHostCopy);
EVENT(DeviceDeviceCopy);
EVENT(GPUCall);
EVENT(GPUCallEnd);
EVENT(Lock);

STATE(ThreadCreate);
STATE(IORead);
STATE(IOWrite);
STATE(Exclusive);
STATE(Init);

}

using namespace paraver;

/* Macros to issue traces in paraver mode */
#define addThread()	if(paraver::trace != NULL) paraver::trace->__addThread()
#define addThreadTid(t)	if(paraver::trace != NULL) paraver::trace->__addThread(t)
static inline
Time_t pushState(StateName * s)
{
    return paraver::trace->__pushState(*s);
}

static inline
Time_t pushState(StateName * s, uint32_t tid)
{
    return paraver::trace->__pushState(*s, tid);
}

static inline
void pushStateAt(Time_t t, StateName * s)
{
    paraver::trace->__pushState(t, *s);
}

static inline
void pushStateAt(Time_t t, StateName * s, uint32_t tid)
{
    paraver::trace->__pushState(t, *s, tid);
}

static inline
Time_t popState()
{
    return paraver::trace->__popState();
}

static inline
Time_t popState(uint32_t tid)
{
    return paraver::trace->__popState(tid);
}

static inline
void popStateAt(Time_t t)
{
    paraver::trace->__popState(t);
}

static inline
void popStateAt(Time_t t, uint32_t tid)
{
    paraver::trace->__popState(t, tid);
}

static inline
Time_t pushEvent(EventName * e, uint64_t val)
{
    return paraver::trace->__pushEvent(*e, val);
}

static inline
Time_t pushEvent(EventName * e, uint32_t tid, uint64_t val)
{
    return paraver::trace->__pushEvent(*e, tid, val);
}

static inline
void pushEventAt(Time_t t, EventName * e, uint64_t val)
{
    paraver::trace->__pushEvent(*e, val);
}

static inline
void pushEventAt(Time_t t, EventName * e, uint32_t tid, uint64_t val)
{
    paraver::trace->__pushEvent(t, *e, tid, val);
}

#define enterFunction(s) \
	if(paraver::trace != NULL) paraver::trace->__pushEvent(*paraver::Function, paraver::s)
#define exitFunction() \
	if(paraver::trace != NULL) paraver::trace->__pushEvent(*paraver::Function, 0)

#define enterLock(s) \
	if(paraver::trace != NULL) {\
		paraver::trace->__pushState(paraver::trace->__pushEvent(*paraver::Lock, s), *paraver::Exclusive);\
	}
#define exitLock() \
	if(paraver::trace != NULL) {\
		paraver::trace->__popState(paraver::trace->__pushEvent(*paraver::Lock, 0));\
	}

#else

#define addThread()
#define pushState(...)
#define pushStateAt(...)
#define popState(...)
#define popStateAt(...)
#define pushEvent(...)
#define pushEventAt(...)

#define enterFunction(s)
#define exitFunction()

#define enterLock(s)
#define exitLock()

#endif


#endif
