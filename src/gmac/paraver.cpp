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
	"accMalloc", "accFree",
	"accHostDevice", "accDeviceHost", "accDeviceDeviceCopy",
	"accLaunch", "accSync",
	"gmacMalloc", "gmacGlobalMalloc", "gmacFree", "gmacLaunch", "gmacSync", "gmacSignal", "gmacAccs", "gmacSetAffinity", "gmacClear", "gmacBind", "gmacUnbind",
	"vmAlloc", "vmFree", "vmFlush", "vmSync",
	NULL
};

static const char *lockNames[] = {
	"Unlock",
	"mmLocal", "mmGlobal", "mmShared", "pageTable", "ctxLocal", "ctxGlobal", "ctxCreate",
	"queueLock",
	"ioHostLock", "ioDeviceLock",
	"process", "writeMutex", "rollingMap", "rollingBuffer", "manager", "threadQueue", "region",
    "pthread", "segv", "shMap", "contextList", "blockList", "queueMap",
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
