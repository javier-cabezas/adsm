#if defined(USE_TRACE_CONSOLE)
#include "Console.h"

#include <iostream>

namespace __impl { namespace trace {

void InitApiTracer()
{
	tracer = new __impl::trace::Console();
}
void FiniApiTracer()
{
	if(tracer != NULL) delete tracer;
}

Console::Console() :
	os(std::cerr)
{}

Console::~Console()
{}

void Console::startThread(THREAD_T tid, const char *name)
{
	os << "@THREAD:START:" << timeMark() << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::endThread(THREAD_T tid)
{
	os << "@THREAD:END:" << timeMark() << ":" << tid << "@" << std::endl;
}

void Console::enterFunction(THREAD_T tid, const char *name)
{
	os << "@FUNCTION:START:" << timeMark() << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::exitFunction(THREAD_T tid, const char *name)
{
	os << "@FUNCTION:END:" << timeMark() << ":" << tid << ":" << name << "@" << std::endl;
}

void Console::setThreadState(THREAD_T tid, State state)
{
	os << "@STATE:" << timeMark() << ":" << tid << ":" << state << "@" << std::endl;
}

}}

#endif
