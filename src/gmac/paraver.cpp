#include <gmac/paraver.h>

#ifdef PARAVER

#include <paraver/Trace.h>
#include <paraver/Names.h>


namespace paraver {

extern Trace *trace;

void __pushState(paraver::StateName &s) { if(trace != NULL) trace->__pushState(s); }
void __popState(void) { if(trace != NULL) trace->__popState(); }
void __pushEvent(paraver::EventName &e, int v) { if(trace != NULL) trace->__pushEvent(e, v); }

int __init_paraver = 0;

EVENT_IMPL(Function);
EVENT_IMPL(HostDeviceCopy);
EVENT_IMPL(DeviceHostCopy);
EVENT_IMPL(DeviceDeviceCopy);
EVENT_IMPL(GPUCall);
EVENT_IMPL(Lock);

STATE_IMPL(ThreadCreate);
STATE_IMPL(IORead);
STATE_IMPL(IOWrite);
STATE_IMPL(Exclusive);
STATE_IMPL(Init);

};

static const char *functionNames[] = {
	"None",                
	"accMalloc",        // 1
    "accFree",          // 2
	"accHostDevice",    // 3
    "accDeviceHost",    // 4
    "accDeviceDevice",  // 5
	"accLaunch",        // 6
    "accSync",          // 7
	"gmacMalloc",       // 8
    "gmacGlobalMalloc", // 9
    "gmacFree",         // 10
    "gmacLaunch",       // 11
    "gmacSync",         // 12
    "gmacSignal",       // 13
    "gmacAccs",         // 14
    "gmacSetAffinity",  // 15
    "gmacClear",        // 16
    "gmacBind",         // 17
    "gmacUnbind",       // 18
	"vmAlloc",          // 19
    "vmFree",           // 20
    "vmFlush",          // 21
    "vmSync",           // 22
	NULL
};

static const char *lockNames[] = {
	"Unlock",
	"mmLocal",       // 1
    "mmGlobal",      // 2
    "mmShared",      // 3
    "pageTable",     // 4
    "ctxLocal",      // 5
    "ctxGlobal",     // 6
    "ctxCreate",     // 7
	"queueLock",     // 8
	"ioHostLock",    // 9
    "ioDeviceLock",  // 10
	"process",       // 11
    "writeMutex",    // 12
    "rollingMap",    // 13
    "rollingBuffer", // 14
    "manager",       // 15
    "threadQueue",   // 16
    "region",        // 17
    "pthread",       // 18
    "shMap",         // 19
    "contextList",   // 20
    "blockList",     // 21
    "queueMap",      // 22
	NULL
};


void paraverInit(void)
{
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::Function->registerType(i, std::string(functionNames[i]));
	for(int i = 0; lockNames[i] != NULL; i++)
		paraver::Lock->registerType(i, std::string(lockNames[i]));
}


#endif
