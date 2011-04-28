#if defined(USE_TRACE)
#include "trace/Tracer.h"

namespace __impl { namespace trace {
Tracer *tracer = NULL;

Atomic threads_;
util::Private<int32_t> tid_;

CONSTRUCTOR(InitTracer);

static void InitTracer()
{
#if defined(USE_TRACE)
    util::Private<int32_t>::init(tid_);
	InitApiTracer();
    SetThreadState(Running);
#endif
}

DESTRUCTOR(FiniTracer);
static void FiniTracer()
{
#if defined(USE_TRACE)
	FiniApiTracer();
#endif
}


}}
#endif
