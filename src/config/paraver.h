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
	accMalloc = 1, accFree,
	accHostDeviceCopy, accDeviceHostCopy, accDeviceDeviceCopy,
	accLaunch, accSync,
	gmacMalloc, gmacGlobalMalloc, gmacFree, gmacLaunch, gmacSync, gmacSignal, gmacAccs, gmacSetAffinity, gmacClear, gmacBind, gmacUnbind,
	vmAlloc, vmFree, vmFlush, vmSync,
};

enum LockName {
	mmLocal = 1, mmGlobal, mmShared, pageTable, ctxLocal, ctxGlobal, ctxCreate,
	queueLock, 
	ioHostLock, ioDeviceLock,
	process, writeMutex, rollingMap, rollingBuffer, manager, queue, region,
	pthread, segv, shMap, contextList
};
};

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
EVENT(Lock);

STATE(ThreadCreate);
STATE(IORead);
STATE(IOWrite);
STATE(Exclusive);
STATE(Init);


};

/* Macros to issue traces in paraver mode */
#define addThread()	if(paraver::trace != NULL) paraver::trace->__addThread()
#define pushState(s)	if(paraver::trace != NULL) paraver::trace->__pushState(*paraver::s)
#define popState()	if(paraver::trace != NULL) paraver::trace->__popState()
#define pushEvent(e, ...)\
	if(paraver::trace != NULL) paraver::trace->__pushEvent(*paraver::e, ##__VA_ARGS__)

#define enterFunction(s) \
	if(paraver::trace != NULL) paraver::trace->__pushEvent(*paraver::Function, paraver::s)
#define exitFunction() \
	if(paraver::trace != NULL) paraver::trace->__pushEvent(*paraver::Function, 0)

#define enterLock(s) \
	if(paraver::trace != NULL) {\
		paraver::trace->__pushEvent(*paraver::Lock, s);\
		paraver::trace->__pushState(*paraver::Exclusive);\
	}
#define exitLock() \
	if(paraver::trace != NULL) {\
		paraver::trace->__pushEvent(*paraver::Lock, 0);\
		paraver::trace->__popState();\
	}

#else

#define addThread()
#define pushState(s)
#define popState()
#define pushEvent(e)

#define enterFunction(s)
#define exitFunction()

#define enterLock(s)
#define exitLock()

#endif


#endif
