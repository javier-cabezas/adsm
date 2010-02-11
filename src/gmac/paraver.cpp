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

EVENT_IMPL(Function, 1);
EVENT_IMPL(HostDeviceCopy, 2);
EVENT_IMPL(DeviceHostCopy, 3);
EVENT_IMPL(DeviceDeviceCopy, 4);
EVENT_IMPL(GPUCall, 5);
EVENT_IMPL(Lock, 6);

STATE_IMPL(ThreadCreate, 2);
STATE_IMPL(IORead, 3);
STATE_IMPL(IOWrite, 4);
STATE_IMPL(Exclusive, 5);
STATE_IMPL(Init, 6);

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
	"mmLocal", "mmGlobal", "pageTable", "ctxLocal", "ctxGlobal",
	"queueLock",
	"ioHostLock", "ioDeviceLock",
	"process", "writeMutex", "rollingBuffer", "manager",
	NULL
};


static void __attribute__((constructor(199))) paraverInit(void)
{
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::Function->registerType(i, std::string(functionNames[i]));
	for(int i = 0; lockNames[i] != NULL; i++)
		paraver::Lock->registerType(i, std::string(lockNames[i]));
}


#endif
