#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include "memory/Manager.h"

#include "gmac/init.h"

#include <debug.h>

gmac::Process *proc = NULL;

namespace gmac {

ContextList::ContextList() :
    RWLock(paraver::contextList)
{}

ThreadQueue::ThreadQueue() :
    hasContext(paraver::queue)
{}

size_t Process::_totalMemory = 0;


Process::Process() :
    mutex(paraver::process), current(0)
{}

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
    for(q = _queues.begin(); q != _queues.end(); q++) {
        if (q->second->queue != NULL) {
            delete q->second->queue;
        }
        delete q->second;
    }
    _accs.clear();
    mutex.unlock();
    memoryFini();
}

void
Process::init(const char *name)
{
    // Process is a singleton class. The only allowed instance is proc
    TRACE("Initializing process");
    ASSERT(proc == NULL);
    Context::Init();
    proc = new Process();
    apiInit();
    memoryInit(name);
    // Register first, implicit, thread
    proc->initThread();
    gmac::Context::initThread();
}

void
Process::initThread()
{
    ThreadQueue * q = new ThreadQueue();
    q->hasContext.lock();
    q->queue = NULL;
    mutex.lock();
    _queues.insert(QueueMap::value_type(SELF(), q));
    mutex.unlock();
}

Context *
Process::create(int acc)
{
    pushState(Init);
    TRACE("Creating new context");
    mutex.lock();
    QueueMap::iterator q = _queues.find(SELF());
    ASSERT(q != _queues.end());
    Context * ctx;
    int usedAcc;

    if (acc != ACC_AUTO_BIND) {
        ASSERT(acc < _accs.size());
        usedAcc = acc;
        ctx = _accs[acc]->create();
    } else {
        // Bind the new Context to the accelerator with less contexts
        // attached to it
        usedAcc = 0;
        for (int i = 1; i < _accs.size(); i++) {
            if (_accs[i]->nContexts() < _accs[usedAcc]->nContexts()) {
                usedAcc = i;
            }
        }

        ctx = _accs[usedAcc]->create();
        _contexts.push_back(ctx);
    }
    q->second->queue = new Queue();
    q->second->hasContext.unlock();
	mutex.unlock();
	TRACE("Created context on Acc#%d", usedAcc);
    popState();
    return ctx;
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
        _accs[acc]->create();
    }
	TRACE("Migrated context");
	mutex.unlock();
    return ret;
}


void Process::remove(Context *ctx)
{
	mutex.lock();
	_contexts.remove(ctx);
    ThreadQueue * q = _queues[SELF()];
    if (q->queue != NULL) {
        delete q->queue;
    }
    delete q;
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
    Context * ctx = Context::current();
    mutex.lock();
	QueueMap::iterator q = _queues.find(id);
    mutex.unlock();
	ASSERT(q != _queues.end());
    q->second->hasContext.lock();
    q->second->hasContext.unlock();
	q->second->queue->push(ctx);
	PRIVATE_SET(Context::key, NULL);
	q = _queues.find(SELF());
	ASSERT(q != _queues.end());
	PRIVATE_SET(Context::key, q->second->queue->pop());
}

}
