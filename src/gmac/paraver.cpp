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

STATE_IMPL(ThreadCreate, 2);
STATE_IMPL(IORead, 3);
STATE_IMPL(IOWrite, 4);

};

static const char *functionNames[] = {
	"None",
	"accMalloc", "accFree",
	"accHostDevice", "accDeviceHost", "accDeviceDeviceCopy",
	"accLaunch", "accSync",
	"gmacMalloc", "gmacFree", "gmacLaunch", "gmacSync", "gmacSignal",
	NULL
};

static void __attribute__((constructor(199))) paraverInit(void)
{
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::Function->registerType(i, std::string(functionNames[i]));
}


#endif
