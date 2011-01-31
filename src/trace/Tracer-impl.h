#ifndef GMAC_TRACE_TRACER_IMPL_H_
#define GMAC_TRACE_TRACER_IMPL_H_

#include "util/Thread.h"

namespace __impl { namespace trace {

#if defined(USE_TRACE)
extern Tracer *tracer;

void InitApiTracer();
void FiniApiTracer();

inline
Tracer::Tracer() : base_(0)
{
	base_ = timeMark();
}

inline
Tracer::~Tracer()
{
}

#endif

inline void InitTracer()
{
#if defined(USE_TRACE)
	InitApiTracer();
#endif
}

inline void FiniTracer()
{
#if defined(USE_TRACE)
	FiniApiTracer();
#endif
}

inline void StartThread(THREAD_T tid, const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) {		
		tracer->startThread(tid, name);
		tracer->setThreadState(tid, Init);
	}
#endif
}

inline void StartThread(const char *name)
{
#if defined(USE_TRACE)
	return StartThread(util::GetThreadId(), name);
#endif
}

inline void EndThread(THREAD_T tid)
{
#if defined(USE_TRACE)	
	if(tracer != NULL) tracer->endThread(tid);
#endif
}

inline void EndThread()
{
#if defined(USE_TRACE)
	return EndThread(util::GetThreadId());
#endif
}

inline void EnterFunction(THREAD_T tid, const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->enterFunction(tid, name);
#endif
}

inline void EnterFunction(const char *name)
{
#if defined(USE_TRACE)
	return EnterFunction(util::GetThreadId(), name);
#endif
}

inline void ExitFunction(THREAD_T tid, const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->exitFunction(tid, name);
#endif
}

inline void ExitFunction(const char *name)
{
#if defined(USE_TRACE)
	return ExitFunction(util::GetThreadId(),name);
#endif
}

inline void RequestLock(const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->requestLock(util::GetThreadId(), name);
#endif
}

inline void AcquireLockExclusive(const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->acquireLockExclusive(util::GetThreadId(), name);
#endif
}

inline void AcquireLockShared(const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->acquireLockShared(util::GetThreadId(), name);
#endif
}

inline void ExitLock(const char *name)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->exitLock(util::GetThreadId(), name);
#endif
}

inline void SetThreadState(THREAD_T tid, const State &state)
{
#if defined(USE_TRACE)
	if(tracer != NULL) tracer->setThreadState(tid, state);
#endif
}

inline void SetThreadState(const State &state)
{
#if defined(USE_TRACE)
	return SetThreadState(util::GetThreadId(), state);
#endif
}

inline void DataCommunication(THREAD_T src, THREAD_T dst, uint64_t delta, size_t size)
{
#if defined(USE_TRACE)
    if(tracer != NULL) tracer->dataCommunication(src, dst, delta, size);
#endif
}

inline void DataCommunication(THREAD_T tid, uint64_t delta, size_t size)
{
#if defined(USE_TRACE)
    return DataCommunication(util::GetThreadId(), tid, delta, size);
#endif
}

inline void TimeMark(uint64_t &mark)
{
#if defined(USE_TRACE)
    if(tracer != NULL) mark = tracer->timeMark();
    else mark = 0;
#endif
}

}}

#endif
