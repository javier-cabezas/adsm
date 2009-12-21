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
	QueueMap::iterator q;
	mutex.lock();
	for(c = _contexts.begin(); c != _contexts.end(); c++) {
		(*c)->destroy();
	}
	for(a = _accs.begin(); a != _accs.end(); a++)
		delete *a;
    for(q = _queues.begin(); q != _queues.end(); q++)
        delete q->second;
	_accs.clear();
	mutex.unlock();
	memoryFini();
}

void Process::create()
{
	TRACE("Creating new context");
	mutex.lock();
	unsigned n = current;
	current = ++current % _accs.size();
	Context *ctx = _accs[n]->create();
	ctx->init();
	_contexts.push_back(ctx);
	_queues.insert(QueueMap::value_type(SELF(), new kernel::Queue()));
	mutex.unlock();
}

void Process::clone(gmac::Context *ctx, int acc)
{
	TRACE("Cloning context");
	mutex.lock();
	unsigned n = current;
	current = ++current % _accs.size();
	Context *clon = _accs[n]->clone(*ctx);
	clon->init();
	_contexts.push_back(clon);
	_queues.insert(QueueMap::value_type(SELF(), new kernel::Queue()));
	mutex.unlock();
	TRACE("Cloned context on Acc#%d", n);
}

gmacError_t Process::migrate(int acc)
{
	mutex.lock();
    if (acc >= _accs.size()) return gmacErrorInvalidValue;
    gmacError_t ret = gmacSuccess;
	TRACE("Migrating context");
    if (Context::hasCurrent()) {
        // Really migrate data
        abort();
    } else {
        // Create the context in the requested accelerator
        Context::create(acc);
    }
	TRACE("Migrated context");
	mutex.unlock();
    return ret;
}


void Process::remove(Context *ctx)
{
	mutex.lock();
	_contexts.remove(ctx);
    delete _queues[SELF()];
	_queues.erase(SELF());
	mutex.unlock();
	ctx->destroy();
}

void Process::accelerator(Accelerator *acc) 
{
	_accs.push_back(acc);
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
	q->second->push(gmac::Context::current());
	PRIVATE_SET(Context::key, NULL);
	q = _queues.find(SELF());
	assert(q != _queues.end());
	PRIVATE_SET(Context::key, q->second->pop());
}


}
