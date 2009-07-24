#include "Trace.h"

#include <config/debug.h>

paraver::Trace *trace = NULL;

static void __attribute__((constructor(199))) paraverInit(void)
{
	TRACE("Paraver Tracing");
	trace = new paraver::Trace("paraver.prb");
}

static void __attribute__((destructor)) paraverFini(void)
{
	trace->write();
	delete trace;
}
