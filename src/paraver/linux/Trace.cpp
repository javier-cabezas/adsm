#include <paraver/Trace.h>
#include <paraver/Element.h>

#include <common/threads.h>
#include <common/debug.h>

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
	MUTEX_INIT(ofMutex);

	// Create the root application and add the current task
	apps.push_back(new Application(1, "app"));
	Task *task = apps.back()->addTask(getpid());
	task->addThread(gettid());
	pushState(State::Running);
}

void Trace::pushState(unsigned state)
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(gettid());
	Time_t timeStamp = getTimeStamp();

	MUTEX_LOCK(ofMutex);
	thread->start(of, state, timeStamp);
	MUTEX_UNLOCK(ofMutex);
}

void Trace::popState()
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(gettid());
	Time_t timeStamp = getTimeStamp();

	MUTEX_LOCK(ofMutex);
	thread->end(of, timeStamp);
	MUTEX_UNLOCK(ofMutex);
}

void Trace::event(unsigned type, unsigned value)
{
	Task *task = apps.back()->getTask(gettid());
	Time_t timeStamp = getTimeStamp();

	Event event(task->getThread(gettid()), timeStamp, type, value);

	MUTEX_LOCK(ofMutex);
	event.write(of);
	MUTEX_UNLOCK(ofMutex);
}

};
