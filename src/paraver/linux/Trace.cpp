#include "cpu.h"
#include <paraver/Trace.h>

#include <strings.h>
#include <assert.h>

namespace paraver {

Trace::Trace(const char *fileName) :
	pendingTime(0)
{
	// Create the current node and the cpu
	nodes.push_back(new Node(0, "root"));
	int nCpus = sysconf(_SC_NPROCESSORS_ONLN);
	for(int n = 0; n < nCpus; n++) nodes.back()->addCpu(n);

	// Create the root application and add the current task
	apps.push_back(new Application(0, "app"));
	apps.back()->addTask(nodes.back(), getpid());

	// Init the output file and the communication counter
	of.open(fileName, std::ios::out);
}

void Trace::pushState(unsigned state)
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(sched_getcpu());
	Time_t timeStamp = getTime();
	thread->start(state, timeStamp);

	State &st = thread->start(state, timeStamp);
	st.write(of);
	
	setEnd(timeStamp);
}

void Trace::popState()
{
	Task *task = apps.back()->getTask(getpid());
	Thread *thread = task->getThread(sched_getcpu());
	Time_t timeStamp = getTime();

	State &st = thread->end(timeStamp);
	st.write(of);
	
	setEnd(timeStamp);
}

void Trace::event(unsigned type, unsigned value)
{
	Task *task = apps.back()->getTask(getpid());
	Time_t timeStamp = getTime();

	Event event(task->getThread(sched_getcpu()), timeStamp, type, value);
	event.write(of);
	setEnd(timeStamp);
}

};
