#include <gmac/paraver.h>

#ifdef PARAVER

#include <paraver/Trace.h>

extern paraver::Trace *trace;

void pushState(paraver::StateName &s) { trace->pushState(s); }
void popState(void) { trace->popState(); }
void pushEvent(paraver::EventName &e, int v) { trace->pushEvent(e, v); }

#endif
