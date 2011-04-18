#if defined(USE_TRACE)
#include "trace/Tracer.h"

namespace __impl { namespace trace {
Tracer *tracer = NULL;

Atomic threads_;
util::Private<int32_t> tid_;

void CONSTRUCTOR InitTracer()
{
#if defined(USE_TRACE)
    util::Private<int32_t>::init(tid_);
	InitApiTracer();
    SetThreadState(Running);
#endif
}

void DESTRUCTOR FiniTracer()
{
#if defined(USE_TRACE)
	FiniApiTracer();
#endif
}


}}
#endif
