#include <gmac/paraver.h>

#ifdef PARAVER

#include <paraver/Trace.h>

extern paraver::Trace *trace;

void pushState(paraver::StateName &s) { trace->__pushState(s); }
void popState(void) { trace->__popState(); }
void pushEvent(paraver::EventName &e, int v) { trace->__pushEvent(e, v); }

#endif
