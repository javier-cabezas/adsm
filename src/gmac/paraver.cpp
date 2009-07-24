#include <gmac/paraver.h>

#ifdef PARAVER

#include <paraver/Trace.h>

extern paraver::Trace *trace;

void pushState(paraver::StateName &s) { if(trace != NULL) trace->__pushState(s); }
void popState(void) { if(trace != NULL) trace->__popState(); }
void pushEvent(paraver::EventName &e, int v) { if(trace != NULL) trace->__pushEvent(e, v); }

#endif
