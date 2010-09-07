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

EVENT_IMPL(HostDeviceCopy);
EVENT_IMPL(DeviceHostCopy);
EVENT_IMPL(DeviceDeviceCopy);
EVENT_IMPL(Accelerator);

STATE_IMPL(ThreadCreate);
STATE_IMPL(IORead);
STATE_IMPL(IOWrite);
STATE_IMPL(Init);

};


static const char *AcceleratorNames[] = {
	"None",
	"AcceleratorRun", // 1
    "AcceleratorIO",  // 2
	NULL
};


void paraverInit(void)
{
	for(int i = 0; AcceleratorNames[i] != NULL; i++)
		paraver::Accelerator->registerType(__AcceleratorNameBase + i, std::string(AcceleratorNames[i]));
}


#endif
