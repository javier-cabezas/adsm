#if defined(ENABLE_TRACE_CONSOLE)
#include "Console.h"

#include <iostream>

namespace gmac { namespace trace {

void InitApiTracer()
{
	tracer = new Console();
}
void FiniApiTracer()
{
	if(tracer != NULL) delete tracer;
}

Console::Console() :
	of(std::cerr)
{}

Console::~Console()
{}

void Console::startThread(THREAD_T tid)
{
	os << "@THREAD : START : " << timeMark() << " : " << tid << "@" << std::endl;
}

void Console::endThread(THREAD_T tid)
{
	os << "@THREAD : END : " << timeMark() << " : " << tid << "@" << std::endl;
}

void Console::startFunction(THREAD_T tid, const char *name)
{
	os << "@FUNCTION : START : " << timeMark() << " : " << tid << " : " << name << "@" << std::endl;
}

void Console::endFunction(THREAD_T tid, const char *name)
{
	os << "@FUNCTION : END : " << timeMark() << " : " << tid << " : " << name << "@" << std::endl;
}

void Console::setThreadState(THREAD_T tid, State state)
{
	os << "@STATE : END : " << timeMark() << " : " << tid << " : " << state << "@" << std::endl;
}

}}

#endif