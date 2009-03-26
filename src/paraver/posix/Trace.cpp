#include <paraver/threads.h>
#include <paraver/Trace.h>
#include <paraver/Element.h>
#include <paraver/Types.h>

#include <config/debug.h>

#include <strings.h>
#include <assert.h>

namespace paraver {

Trace::Trace(const char *fileName) :
	startTime(getTime()),
	endTime(0),
	pendingTime(0)
{
	// Init the output file and the communication counter
	of.open(fileName, std::ios::out);
	PARAVER_MUTEX_INIT(ofMutex);

	// Create the root application and add the current task
	apps.push_back(new Application(1, "app"));
	Task *task = apps.back()->addTask(getpid());
	task->__addThread(paraver_gettid());
	__pushState(_Running_);
}

void Trace::__pushState(const StateName &state)
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(paraver_gettid());
	Time_t timeStamp = getTimeStamp();

	PARAVER_MUTEX_LOCK(ofMutex);
	thread->start(of, state.getValue(), timeStamp);
	PARAVER_MUTEX_UNLOCK(ofMutex);
}

void Trace::__popState()
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(paraver_gettid());
	Time_t timeStamp = getTimeStamp();

	PARAVER_MUTEX_LOCK(ofMutex);
	thread->end(of, timeStamp);
	PARAVER_MUTEX_UNLOCK(ofMutex);
}

void Trace::__pushEvent(const EventName &ev, int value)
{
	Task *task = apps.back()->getTask(getpid());
	Time_t timeStamp = getTimeStamp();

	Event event(task->getThread(paraver_gettid()), timeStamp, ev.getValue(), value);

	PARAVER_MUTEX_LOCK(ofMutex);
	event.write(of);
	PARAVER_MUTEX_UNLOCK(ofMutex);
}

};
