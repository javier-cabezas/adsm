#include "Trace.h"

paraver::Trace *trace;

static void __attribute__((constructor)) paraverInit(void)
{
	trace = new paraver::Trace("paraver.prb");
}

static void __attribute__((destructor)) paraverFini(void)
{
	delete trace;
}
