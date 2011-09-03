#if defined(USE_TRACE)
#include "trace/Tracer.h"

namespace __impl { namespace trace {
Tracer *tracer = NULL;

Atomic threads_;
PRIVATE int32_t tid_ = TID_INVALID;

CONSTRUCTOR(init);
static void init()
{
#if defined(USE_TRACE)
	InitApiTracer();
    SetThreadState(Running);
#endif
}

DESTRUCTOR(fini);
static void fini()
{
#if defined(USE_TRACE)
	FiniApiTracer();
#endif
}


}}
#endif
