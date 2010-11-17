#ifndef GMAC_TRACE_TRACER_IMPL_H_
#define GMAC_TRACE_TRACER_IMPL_H_

#include "util/Thread.h"

namespace gmac { namespace trace {

#if defined(ENABLE_TRACE)
Trace *trace = NULL;

void InitApiTracer();
void FiniApiTracer();

#endif

inline void InitTracer()
{
#if defined(ENABLE_TRACE)
	InitApiTracer();
#endif
}

inline void FiniTracer()
{
#if defined(ENABLE_TRACE)
	FiniApiTracer();
#endif
}

inline void StartThread(THREAD_T tid)
{
#if defined(ENABLE_TRACE)
	if(trace != NULL) trace->startThread(tid);
#endif
}

inline void StartThread()
{
#if defined(ENABLE_TRACE)
	return StartThread(util::GetThreadId());
#endif
}

inline void EndThread(THREAD_T tid)
{
#if defined(ENABLE_TRACE)	
	if(trace != NULL) trace->endThread(tid);
#endif
}

inline void EndThread()
{
#if defined(ENABLE_TRACE)
	return EndThread(util::GetThreadId());
#endif
}

inline void EnterFunction(THREAD_T tid, const char *name)
{
#if defined(ENABLE_TRACE)
	if(trace != NULL) trace->enterFunction(tid, name);
#endif
}

inline void EnterFunction(const char *name)
{
#if defined(ENABLE_TRACE)
	return EnterFunction(util::GetThreadId(), name);
#endif
}

inline void ExitFunction(THREAD_T tid, const char *name)
{
#if defined(ENABLE_TRACE)
	if(trace != NULL) trace->exitFunction(tid, name);
#endif
}

inline void ExitFunction(const char *name)
{
#if defined(ENABLE_TRACE)
	return ExitFunction(util::GetThreadId(),name);
#endif
}

inline void SetThreadState(THREAD_T tid, const State &state)
{
#if defined(ENABLE_TRACE)
	if(trace != NULL) trace->setThreadState(tid, state);
#endif
}

inline void SetThreadState(const State &state)
{
#if defined(ENABLE_TRACE)
	return SetThreadState(util::GetThreadId(), state);
#endif
}

}}

#endif
