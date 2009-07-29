#include "Trace.h"

#include <config/debug.h>

paraver::Trace *trace = NULL;
static const char *paraverVar = "PARAVER_OUTPUT";
static const char *defaultOut = "paraver.prb";

static void __attribute__((constructor(199))) paraverInit(void)
{
	TRACE("Paraver Tracing");
	const char *file = getenv(paraverVar);
	if(file == NULL) file = defaultOut;
	trace = new paraver::Trace(file);
}

static void __attribute__((destructor)) paraverFini(void)
{
	trace->write();
	delete trace;
}
