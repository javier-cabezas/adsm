#include "Trace.h"

#include <common/debug.h>

paraver::Trace *trace;

static void __attribute__((constructor)) paraverInit(void)
{
	TRACE("Paraver Tracing");
	trace = new paraver::Trace("paraver.prb");
}

static void __attribute__((destructor)) paraverFini(void)
{
	trace->write();
	delete trace;
}
