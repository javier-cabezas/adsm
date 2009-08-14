#include "Trace.h"
#include "Pcf.h"

#include <config/debug.h>

paraver::Trace *trace = NULL;
static const char *paraverVar = "PARAVER_OUTPUT";
static const char *defaultOut = "paraver";

static void __attribute__((constructor(199))) paraverInit(void)
{
	TRACE("Paraver Tracing");
	const char *__file = getenv(paraverVar);
	if(__file == NULL) __file = defaultOut;
	std::string file = std::string(__file) + ".prb";
	trace = new paraver::Trace(file.c_str());
}

static void __attribute__((destructor(199))) paraverFini(void)
{
	const char *__file = getenv(paraverVar);
	if(__file == NULL) __file = defaultOut;
	std::string file = std::string(__file) + ".pcf";
	std::ofstream of(file.c_str(), std::ios::out);
	paraver::pcf(of);
	of.close();

	paraver::StateName::destroy();
	paraver::EventName::destroy();

	trace->write();
	delete trace;
}
