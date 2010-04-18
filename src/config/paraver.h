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
    __FunctionNameBase   = 1000,
    FuncAccMalloc        = 1001,  // 1
    FuncAccFree          = 1002,  // 2
    FuncAccHostDevice    = 1003,  // 3
    FuncAccDeviceHost    = 1004,  // 4
    FuncAccDeviceDevice  = 1005,  // 5
    FuncAccLaunch        = 1006,  // 6
    FuncAccSync          = 1007,  // 7
    FuncGmacMalloc       = 1008,  // 8
    FuncGmacGlobalMalloc = 1009,  // 9
    FuncGmacFree         = 1010, // 10
    FuncGmacLaunch       = 1011, // 11
    FuncGmacSync         = 1012, // 12
    FuncGmacSignal       = 1013, // 13
    FuncGmacAccs         = 1014, // 14
    FuncGmacSetAffinity  = 1015, // 15
    FuncGmacClear        = 1016, // 16
    FuncGmacBind         = 1017, // 17
    FuncGmacUnbind       = 1018, // 18
    FuncVmAlloc          = 1019, // 19
    FuncVmFree           = 1020, // 20
    FuncVmFlush          = 1021, // 21
    FuncVmSync           = 1022  // 22
};

enum LockName {
    __LockNameBase   = 2000,
    LockMmLocal      = 2001,  // 1
    LockMmGlobal     = 2002,  // 2
    LockMmShared     = 2003,  // 3
    LockPageTable    = 2004,  // 4
    LockCtxLocal     = 2005,  // 5
    LockCtxGlobal    = 2006,  // 6
    LockCtxCreate    = 2007,  // 7
    LockQueue        = 2008,  // 8
    LockIoHost       = 2009,  // 9
    LockIoDevice     = 2010, // 10
    LockProcess      = 2011, // 11
    LockWriteMutex   = 2012, // 12
    LockRollingMap   = 2013, // 13
    LockRollingBuffer= 2014, // 14
    LockManager      = 2015, // 15
    LockThreadQueue  = 2016, // 16
    LockRegion       = 2017, // 17
    LockPthread      = 2018, // 18
    LockShMap        = 2019, // 19
    LockContextList  = 2020, // 20
    LockBlockList    = 2021, // 21
    LockQueueMap     = 2022, // 22
    LockReference    = 2023, // 23
    LockContext      = 2024  // 24
};

enum AcceleratorName {
    __AcceleratorNameBase = 3000,
    AcceleratorRun = 3001,        // 1
    AcceleratorIO  = 3002         // 2
};
}

using namespace paraver;

#ifdef PARAVER
#include <paraver/Trace.h>
#include <paraver/Names.h>

namespace paraver {

extern Trace *trace;

EVENT(Function);
EVENT(HostDeviceCopy);
EVENT(DeviceHostCopy);
EVENT(DeviceDeviceCopy);
EVENT(Accelerator);
EVENT(Lock);

STATE(ThreadCreate);
STATE(IORead);
STATE(IOWrite);
STATE(Exclusive);
STATE(Init);

}

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
void pushEventState(StateName * s, EventName * e, uint64_t val)
{
    Time_t t = paraver::trace->__pushState(*s);
    paraver::trace->__pushEvent(t, *e, val);
}

static inline
void pushEventState(StateName * s, EventName * e, uint32_t tid, uint64_t val)
{
    Time_t t = paraver::trace->__pushState(*s, tid);
    paraver::trace->__pushEvent(t, *e, tid, val);
}

static inline
void popEventState(EventName * e)
{
    Time_t t = paraver::trace->__popState();
    paraver::trace->__pushEvent(t, *e, 0);
}

static inline
void popEventState(EventName * e, uint32_t tid)
{
    Time_t t = paraver::trace->__popState(tid);
    paraver::trace->__pushEvent(t, *e, tid, 0);
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
#define addThreadTid(...)
#define pushState(...)
#define popState(...)
#define pushEvent(...)
#define pushEventState(...)
#define popEventState(...)

#define enterFunction(s)
#define exitFunction()

#define enterLock(s)
#define exitLock()

#define Time_t uint64_t

#endif


#endif
