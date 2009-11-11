#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include <debug.h>
#include <gmac/init.h>
#include <memory/Manager.h>

gmac::Process *proc = NULL;

namespace gmac {

size_t Process::_totalMemory = 0;

Process::~Process()
{
	TRACE("Cleaning process");
	std::vector<Accelerator *>::iterator a;
	std::list<Context *>::iterator c;
	lock();
	for(c = _contexts.begin(); c != _contexts.end(); c++) {
		(*c)->destroy();
	}
	for(a = accs.begin(); a != accs.end(); a++)
		delete *a;
	accs.clear();
	unlock();
	memoryFini();
}

void Process::create()
{
	TRACE("Creating new context");
	lock();
	unsigned n = current;
	current = ++current % accs.size();
	Context *ctx = accs[n]->create();
	ctx->init();
	_contexts.push_back(ctx);
	_queues.insert(QueueMap::value_type(SELF(), kernel::Queue()));
	unlock();
}

void Process::clone(gmac::Context *ctx)
{
	TRACE("Cloning context");
	lock();
	unsigned n = current;
	current = ++current % accs.size();
	Context *clon = accs[n]->clone(*ctx);
	clon->init();
	_contexts.push_back(clon);
	_queues.insert(QueueMap::value_type(SELF(), kernel::Queue()));
	unlock();
	TRACE("Cloned context on Acc#%d", n);
}

void Process::remove(Context *ctx)
{
	lock();
	_contexts.remove(ctx);
	_queues.erase(SELF());
	unlock();
	ctx->destroy();
}

void Process::accelerator(Accelerator *acc) 
{
	accs.push_back(acc);
	_totalMemory += acc->memory();
}

void *Process::translate(void *addr) 
{
	void *ret = NULL;
	std::list<Context *>::const_iterator i;
	for(i = _contexts.begin(); i != _contexts.end(); i++) {
		ret = (*i)->mm().pageTable().translate(addr);
		if(ret != NULL) return ret;
	}
	return ret;
}

void Process::sendReceive(THREAD_ID id)
{
	QueueMap::iterator q = _queues.find(id);
	assert(q != _queues.end());
	q->second.push(gmac::Context::current());
	PRIVATE_SET(Context::key, NULL);
	q = _queues.find(SELF());
	assert(q != _queues.end());
	PRIVATE_SET(Context::key, q->second.pop());
}


}
