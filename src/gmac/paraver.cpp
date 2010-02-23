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
EVENT_IMPL(GPURun);
EVENT_IMPL(GPUIO);
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
	"LockIoHostLock",    // 9
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
    "LockContextList",   // 20
    "LockBlockList",     // 21
    "LockQueueMap",      // 22
	NULL
};

static const char *gpuRunNames[] = {
	"None",
	"GPURunStart", // 1
    "GPURunEnd",   // 2
	NULL
};

static const char *gpuIONames[] = {
	"None",
	"GPUIOStart", // 1
    "GPUIOEnd",   // 2
	NULL
};


void paraverInit(void)
{
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::Function->registerType(i, std::string(functionNames[i]));
	for(int i = 0; lockNames[i] != NULL; i++)
		paraver::Lock->registerType(i, std::string(lockNames[i]));
	for(int i = 0; gpuRunNames[i] != NULL; i++)
		paraver::GPURun->registerType(i, std::string(gpuRunNames[i]));
	for(int i = 0; gpuIONames[i] != NULL; i++)
		paraver::GPUIO->registerType(i, std::string(gpuIONames[i]));
}


#endif
