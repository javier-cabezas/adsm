#include <config/paraver.h>

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
EVENT_IMPL(Accelerator);
EVENT_IMPL(Lock);

STATE_IMPL(ThreadCreate);
STATE_IMPL(IORead);
STATE_IMPL(IOWrite);
STATE_IMPL(Exclusive);
STATE_IMPL(Init);

};

static const char *functionNames[] = {
	"None",                
	"FuncAccMalloc",        // 1
    "FuncAccFree",          // 2
	"FuncAccHostDevice",    // 3
    "FuncAccDeviceHost",    // 4
    "FuncAccDeviceDevice",  // 5
	"FuncAccLaunch",        // 6
    "FuncAccSync",          // 7
	"FuncGmacMalloc",       // 8
    "FuncGmacGlobalMalloc", // 9
    "FuncGmacFree",         // 10
    "FuncGmacLaunch",       // 11
    "FuncGmacSync",         // 12
    "FuncGmacSignal",       // 13
    "FuncGmacAccs",         // 14
    "FuncGmacSetAffinity",  // 15
    "FuncGmacClear",        // 16
    "FuncGmacBind",         // 17
    "FuncGmacUnbind",       // 18
	"FuncVmAlloc",          // 19
    "FuncVmFree",           // 20
    "FuncVmFlush",          // 21
    "FuncVmSync",           // 22
	NULL
};

static const char *lockNames[] = {
	"Unlock",
	"LockMmLocal",       // 1
    "LockMmGlobal",      // 2
    "LockMmShared",      // 3
    "LockPageTable",     // 4
    "LockCtxLocal",      // 5
    "LockCtxGlobal",     // 6
    "LockCtxCreate",     // 7
	"LockQueueLock",     // 8
	"LockIoLock",        // 9
    "LockIoDeviceLock",  // 10
	"LockProcess",       // 11
    "LockWriteMutex",    // 12
    "LockRollingMap",    // 13
    "LockRollingBuffer", // 14
    "LockManager",       // 15
    "LockThreadQueue",   // 16
    "LockRegion",        // 17
    "LockPthread",       // 18
    "LockShMap",         // 19
    "LockModeMap",       // 20
    "LockBlockList",     // 21
    "LockQueueMap",      // 22
    "LockReference",     // 23
    "LockContext",       // 24
    "LockLog",           // 25
    "LockSystem",        // 26
    "LockObject",        // 27
    "LockBlock",         // 28
	NULL
};

static const char *AcceleratorNames[] = {
	"None",
	"AcceleratorRun", // 1
    "AcceleratorIO",  // 2
	NULL
};


void paraverInit(void)
{
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::Function->registerType(__FunctionNameBase + i, std::string(functionNames[i]));
	for(int i = 0; lockNames[i] != NULL; i++)
		paraver::Lock->registerType(__LockNameBase + i, std::string(lockNames[i]));
	for(int i = 0; AcceleratorNames[i] != NULL; i++)
		paraver::Accelerator->registerType(__AcceleratorNameBase + i, std::string(AcceleratorNames[i]));
}


#endif
