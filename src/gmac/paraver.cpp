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

STATE_IMPL(ThreadCreate);
STATE_IMPL(IORead);
STATE_IMPL(IOWrite);
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
	for(int i = 0; AcceleratorNames[i] != NULL; i++)
		paraver::Accelerator->registerType(__AcceleratorNameBase + i, std::string(AcceleratorNames[i]));
}


#endif
