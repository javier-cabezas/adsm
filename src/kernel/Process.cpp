#include "Process.h"
#include "Context.h"
#include "Accelerator.h"

#include <memory/Manager.h>
#include <gmac/init.h>


gmac::Process *proc = NULL;

namespace gmac {

ThreadQueue::ThreadQueue() :
    util::Lock(LockThreadQueue)
{
    queue = new Queue();
}

ThreadQueue::~ThreadQueue()
{
    delete queue;
}

ContextList::ContextList() :
    RWLock(LockContextList)
{}

QueueMap::QueueMap() : 
    util::RWLock(LockQueueMap)
{}

size_t Process::_totalMemory = 0;

Process::Process() :
    RWLock(LockProcess),
    logger("Process"),
    _global(LockMmGlobal),
    _shared(LockMmShared),
    current(0)
{}

Process::~Process()
{
    logger.trace("Cleaning process");
    std::vector<Accelerator *>::iterator a;
    std::list<Context *>::iterator c;
    QueueMap::iterator q;
    lockWrite();
    while(_contexts.empty() == false)
        _contexts.front()->destroy();
    for(a = _accs.begin(); a != _accs.end(); a++)
        delete *a;
    _accs.clear();
    _queues.lockRead();
    for(q = _queues.begin(); q != _queues.end(); q++) {
        delete q->second;
    }
    _queues.unlock();
    unlock();
    memoryFini();
}

void
Process::init(const char *manager, const char *allocator)
{
    // Process is a singleton class. The only allowed instance is proc
    ::logger->trace("Initializing process");
    ::logger->assertion(proc == NULL);
    Context::Init();
    proc = new Process();
    apiInit();
    memoryInit(manager, allocator);
    // Register first, implicit, thread
    proc->initThread();
    gmac::Context::initThread();
}

void
Process::initThread()
{
    ThreadQueue * q = new ThreadQueue();
    _queues.lockWrite();
    _queues.insert(QueueMap::value_type(SELF(), q));
    _queues.unlock();
}

Context *
Process::create(int acc)
{
    pushState(Init);
    logger.trace("Creating new context");
    lockWrite();
    _queues.lockRead();
    QueueMap::iterator q = _queues.find(SELF());
    logger.assertion(q != _queues.end());
    _queues.unlock();
    Context * ctx;
    unsigned usedAcc;

    if (acc != ACC_AUTO_BIND) {
        logger.assertion(acc < int(_accs.size()));
        usedAcc = acc;
        ctx = _accs[acc]->create();
    } else {
        // Bind the new Context to the accelerator with less contexts
        // attached to it
        usedAcc = 0;
        for (unsigned i = 1; i < _accs.size(); i++) {
            if (_accs[i]->nContexts() < _accs[usedAcc]->nContexts()) {
                usedAcc = i;
            }
        }

        ctx = _accs[usedAcc]->create();
        // Initialize the global shared memory for the context
        manager->initShared(ctx);
        _contexts.push_back(ctx);
    }
	unlock();
	logger.trace("Created context on Acc#%d", usedAcc);
    popState();
    return ctx;
}

gmacError_t Process::migrate(Context * ctx, int acc)
{
	lockWrite();
    if (acc >= int(_accs.size())) return gmacErrorInvalidValue;
    gmacError_t ret = gmacSuccess;
	logger.trace("Migrating context");
    if (Context::hasCurrent()) {
#ifndef USE_MMAP
        if (int(ctx->accId()) != acc) {
            // Create a new context in the requested accelerator
            ret = _accs[acc]->bind(ctx);
        }
#else
        logger.fatal("Migration not implemented when using mmap");
#endif
    } else {
        // Create the context in the requested accelerator
        _accs[acc]->create();
    }
	logger.trace("Context migrated");
	unlock();
    return ret;
}


void Process::remove(Context *ctx)
{
	_contexts.remove(ctx);
}

void Process::addAccelerator(Accelerator *acc)
{
	_accs.push_back(acc);
	_totalMemory += acc->memory();
}

void *Process::translate(void *addr) const
{
	void *ret = NULL;
	std::list<Context *>::const_iterator i;
	for(i = _contexts.begin(); i != _contexts.end(); i++) {
		ret = (*i)->mm().pageTable().translate(addr);
		if(ret != NULL) return ret;
	}
	return ret;
}

void Process::send(THREAD_ID id)
{
    Context *ctx = Context::current();
    _queues.lockRead();
    QueueMap::iterator q = _queues.find(id);
    logger.assertion(q != _queues.end());
    _queues.unlock();
    q->second->lock();
    q->second->queue->push(ctx);
    q->second->unlock();
    Context::key.set(NULL);
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Context *ctx = static_cast<Context *>(Context::key.get());
    if(ctx != NULL) ctx->destroy();
    // Get a fresh context
    _queues.lockRead();
    QueueMap::iterator q = _queues.find(SELF());
    logger.assertion(q != _queues.end());
    _queues.unlock();
    Context::key.set(q->second->queue->pop());
}

void Process::sendReceive(THREAD_ID id)
{
    Context * ctx = Context::current();
    _queues.lockRead();
	QueueMap::iterator q = _queues.find(id);
	logger.assertion(q != _queues.end());
    _queues.unlock();
    q->second->lock();
	q->second->queue->push(ctx);
    q->second->unlock();
    Context::key.set(NULL);
    _queues.lockRead();
	q = _queues.find(SELF());
	logger.assertion(q != _queues.end());
    Context::key.set(q->second->queue->pop());
    _queues.unlock();
}

void Process::copy(THREAD_ID id)
{
    Context *ctx = Context::current();
    _queues.lockRead();
    QueueMap::iterator q = _queues.find(id);
    logger.assertion(q != _queues.end());
    _queues.unlock();
    ctx->inc();
    q->second->lock();
    q->second->queue->push(ctx);
    q->second->unlock();
}

}
