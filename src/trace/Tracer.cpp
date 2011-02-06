#if defined(USE_TRACE)
#include "trace/Tracer.h"

namespace __impl { namespace trace {
Tracer *tracer = NULL;

Atomic threads_;
util::Private<int32_t> tid_;

}}
#endif
