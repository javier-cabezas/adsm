#if defined(USE_TRACE_PARAVER)

#include "Paraver.h"

#include "util/Parameter.h"
#include "paraver/Pcf.h"

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
    baseName_(std::string(util::params::ParamTrace)),
    fileName_(baseName_ + ".trace"),
    trace_(fileName_.c_str(), 1, util::GetThreadId())
{
    FunctionEvent_ = paraver::Factory<paraver::EventName>::create("Function");
    LockEventRequest_ = paraver::Factory<paraver::EventName>::create("LockRequest");
    LockEventAcquireExclusive_ = paraver::Factory<paraver::EventName>::create("LockAcquireExclusive");
    LockEventAcquireShared_ = paraver::Factory<paraver::EventName>::create("LockAcquireShared");
#   define STATE(s) \
        states_.insert(StateMap::value_type(s, paraver::Factory<paraver::StateName>::create(EnumState<s>::name())));
#   include "States-def.h"
#   undef STATE
}

Paraver::~Paraver()
{
    trace_.write(timeMark());

    paraver::TraceReader reader(fileName_.c_str());
    std::string prvFile = baseName_ + ".prv";
    std::ofstream prv(prvFile.c_str());
    prv << reader;
    prv.close();

    std::string pcfFile = baseName_ + ".pcf";
    std::ofstream pcf(pcfFile.c_str());
    paraver::pcf(pcf);
    pcf.close();
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
        id = int32_t(functions_.size() + 1);
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

void Paraver::requestLock(THREAD_T tid, const char *name)
{
    int32_t id = 0;
    int64_t mark = timeMark();
    mutex_.lock();
    LockMap::const_iterator i = locksRequest_.find(std::string(name));
    if(i == locksRequest_.end()) {
        id = int32_t(locksRequest_.size() + 1);
        LockEventRequest_->registerType(id, std::string(name));
        locksRequest_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(mark, 1, tid, *LockEventRequest_, id);
}

void Paraver::acquireLockExclusive(THREAD_T tid, const char *name)
{
    int32_t id = 0;
    uint64_t mark = timeMark();
    trace_.pushEvent(mark, 1, tid, *LockEventRequest_, 0);
    mutex_.lock();
    LockMap::const_iterator i = locksExclusive_.find(std::string(name));
    if(i == locksExclusive_.end()) {
        id = int32_t(locksExclusive_.size() + 1);
        LockEventAcquireExclusive_->registerType(id, std::string(name));
        locksExclusive_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(mark, 1, tid, *LockEventAcquireExclusive_, id);
}

void Paraver::acquireLockShared(THREAD_T tid, const char *name)
{
    int32_t id = 0;
    uint64_t mark = timeMark();
    trace_.pushEvent(mark, 1, tid, *LockEventRequest_, 0);
    mutex_.lock();
    LockMap::const_iterator i = locksShared_.find(std::string(name));
    if(i == locksShared_.end()) {
        id = int32_t(locksShared_.size() + 1);
        LockEventAcquireShared_->registerType(id, std::string(name));
        locksShared_.insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;
    mutex_.unlock();
    trace_.pushEvent(mark, 1, tid, *LockEventAcquireShared_, id);
}

void Paraver::exitLock(THREAD_T tid, const char *name)
{
    uint64_t mark = timeMark();
    mutex_.lock();
    LockMap::const_iterator i = locksExclusive_.find(std::string(name));
    if(i == locksExclusive_.end()) {
        trace_.pushEvent(mark, 1, tid, *LockEventAcquireExclusive_, 0);
    } else {
        ASSERTION(locksShared_.find(std::string(name)) != locksShared.end());
        trace_.pushEvent(mark, 1, tid, *LockEventAcquireShared_, 0);
    }
    mutex_.unlock();
}

void Paraver::setThreadState(THREAD_T tid, const State state)
{
    trace_.pushState(timeMark(), 1, tid, *states_[state]);
}

void Paraver::dataCommunication(THREAD_T src, THREAD_T dst, uint64_t delta, size_t size)
{
    uint64_t current = timeMark();
    trace_.pushCommunication(current - delta, 1, src, current, 1, dst, size);
}

}}

#endif
