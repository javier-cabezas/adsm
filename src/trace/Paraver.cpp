#if defined(USE_TRACE_PARAVER)

#include "Paraver.h"

#include "util/Parameter.h"

namespace __impl { namespace trace {

void InitApiTracer()
{
	tracer = new __impl::trace::Paraver();
}
void FiniApiTracer()
{
	if(tracer != NULL) delete tracer;
}

Paraver::Paraver() :
    trace_(util::params::ParamTrace, 1, util::GetThreadId())
{
    FunctionEvent_ = paraver::Factory<paraver::EventName>::create("Function");
#   define STATE(s) \
        states_.insert(StateMap::value_type(s, paraver::Factory<paraver::StateName>::create(EnumState<s>::name())));
#   include "States-def.h"
#   undef STATE
}

Paraver::~Paraver()
{
}

void Paraver::startThread(THREAD_T tid, const char *name)
{
    trace_.addThread(1, tid);
}

void Paraver::endThread(THREAD_T tid)
{
}

void Paraver::enterFunction(THREAD_T tid, const char *name)
{
    int32_t id = 0;
    mutex_.lock();
    FunctionMap::const_iterator i = functions_.find(std::string(name));
    if(i == functions_.end()) {
        id = int32_t(functions_.size());
        FunctionEvent_->registerType(id, std::string(name));
        functions_.insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(timeMark(), 1, tid, *FunctionEvent_, id);
}

void Paraver::exitFunction(THREAD_T tid, const char *name)
{
    trace_.pushEvent(timeMark(), 1, tid, *FunctionEvent_, 0);
}

void Paraver::setThreadState(THREAD_T tid, const State state)
{
    trace_.pushState(timeMark(), 1, tid, *states_[state]);
}

void Paraver::dataCommunication(THREAD_T src, THREAD_T dst, uint64_t delta, size_t size)
{
}

} }


#endif
