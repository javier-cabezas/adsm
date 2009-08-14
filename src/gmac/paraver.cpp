#include <gmac/paraver.h>

#ifdef PARAVER

#include <paraver/Trace.h>
#include <paraver/Names.h>

extern paraver::Trace *trace;

void pushState(paraver::StateName &s) { if(trace != NULL) trace->__pushState(s); }
void popState(void) { if(trace != NULL) trace->__popState(); }
void pushEvent(paraver::EventName &e, int v) { if(trace != NULL) trace->__pushEvent(e, v); }

int __init_paraver = 0;

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
	fprintf(stderr,"Adding paraver functions\n");
	for(int i = 0; functionNames[i] != 0; i++)
		paraver::_Function_->registerType(i, std::string(functionNames[i]));
}


#endif
