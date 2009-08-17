#include <../threads.h>
#include <../Trace.h>
#include <../Element.h>
#include <paraver/Types.h>

#include <debug.h>

#include <strings.h>
#include <assert.h>

namespace paraver {

Trace::Trace(const char *fileName) :
	startTime(getTime()),
	endTime(0),
	pendingTime(0),
	inTrace(false)
{
	// Init the output file and the communication counter
	of.open(fileName, std::ios::out);
	PARAVER_MUTEX_INIT(ofMutex);

	// Create the root application and add the current task
	apps.push_back(new Application(1, "app"));
	Task *task = apps.back()->addTask(getpid());
	task->__addThread(paraver_gettid());
	__pushState(*_Running_);
}

void Trace::__addThread(void)
{
	return __addThread(getpid(), paraver_gettid());
}

void Trace::__addTask(void)
{
	return __addTask(getpid());
}

void Trace::__pushState(const StateName &state)
{
	return __pushState(getTimeStamp(), getpid(), paraver_gettid(), state);
}

void Trace::__popState()
{
	return __popState(getTimeStamp(), getpid(), paraver_gettid());
}

void Trace::__pushEvent(const EventName &ev, int64_t value)
{
	return __pushEvent(getTimeStamp(), getpid(), paraver_gettid(), ev, value);
}

};
