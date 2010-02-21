#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include "memory/Manager.h"

#include "gmac/init.h"

#include <debug.h>

gmac::Process *proc = NULL;

namespace gmac {

ThreadQueue::ThreadQueue() :
    hasContext(paraver::threadQueue)
{}

ContextList::ContextList() :
    RWLock(paraver::contextList)
{}

QueueMap::QueueMap() : 
    util::RWLock(paraver::queueMap)
{}

size_t Process::_totalMemory = 0;

Process::Process() :
    RWLock(paraver::process), current(0)
{}

Process::~Process()
{
    TRACE("Cleaning process");
    std::vector<Accelerator *>::iterator a;
    std::list<Context *>::iterator c;
    QueueMap::iterator q;
    lockWrite();
    for(c = _contexts.begin(); c != _contexts.end(); c++) {
        (*c)->destroy();
    }
    for(a = _accs.begin(); a != _accs.end(); a++)
        delete *a;
    _accs.clear();
    _queues.lockRead();
    for(q = _queues.begin(); q != _queues.end(); q++) {
        if (q->second->queue != NULL) {
            delete q->second->queue;
        }
        delete q->second;
    }
    _queues.unlock();
    unlock();
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
    _queues.lockWrite();
    _queues.insert(QueueMap::value_type(SELF(), q));
    _queues.unlock();
}

Context *
Process::create(int acc)
{
    pushState(Init);
    TRACE("Creating new context");
    lockWrite();
    _queues.lockRead();
    QueueMap::iterator q = _queues.find(SELF());
    ASSERT(q != _queues.end());
    _queues.unlock();
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
        // Initialize the global shared memory for the context
        manager->initShared(ctx);
        _contexts.push_back(ctx);
    }
    q->second->queue = new Queue();
    q->second->hasContext.unlock();
	unlock();
	TRACE("Created context on Acc#%d", usedAcc);
    popState();
    return ctx;
}

gmacError_t Process::migrate(int acc)
{
	lockWrite();
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
	unlock();
    return ret;
}


void Process::remove(Context *ctx)
{
	_contexts.remove(ctx);
    _queues.lockWrite();
    ThreadQueue * q = _queues[SELF()];
    if (q->queue != NULL) {
        delete q->queue;
    }
    delete q;
	_queues.erase(SELF());
    _queues.unlock();

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
    _queues.lockRead();
	QueueMap::iterator q = _queues.find(id);
	ASSERT(q != _queues.end());
    _queues.unlock();
    q->second->hasContext.lock();
    q->second->hasContext.unlock();
	q->second->queue->push(ctx);
	PRIVATE_SET(Context::key, NULL);
    _queues.lockRead();
	q = _queues.find(SELF());
	ASSERT(q != _queues.end());
	PRIVATE_SET(Context::key, q->second->queue->pop());
    _queues.unlock();
}

}
