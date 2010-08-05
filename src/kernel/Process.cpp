#include "Process.h"
#include "Mode.h"
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

ModeList::ModeList() :
    RWLock(LockModeList)
{}

QueueMap::QueueMap() : 
    util::RWLock(LockQueueMap)
{}

size_t Process::__totalMemory = 0;

Process::Process() :
    RWLock(LockProcess),
    __global(LockMmGlobal),
    __shared(LockMmShared),
    current(0)
{}

Process::~Process()
{
    trace("Cleaning process");
    std::vector<Accelerator *>::iterator a;
    std::list<Mode *>::iterator c;
    QueueMap::iterator q;
    lockWrite();
    while(__modes.empty() == false) {
        __modes.front()->nuke();
        __modes.pop_front();
    }
    for(a = __accs.begin(); a != __accs.end(); a++)
        delete *a;
    __accs.clear();
    __queues.lockRead();
    for(q = __queues.begin(); q != __queues.end(); q++) {
        delete q->second;
    }
    __queues.unlock();
    unlock();
    memoryFini();
}

void
Process::init(const char *manager, const char *allocator)
{
    // Process is a singleton class. The only allowed instance is proc
    util::Logger::TRACE("Initializing process");
    util::Logger::ASSERTION(proc == NULL);
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
    __queues.lockWrite();
    __queues.insert(QueueMap::value_type(SELF(), q));
    __queues.unlock();
}

Mode *
Process::create(int acc)
{
    pushState(Init);
    trace("Creating new execution mode");
    lockWrite();
    __queues.lockRead();
    QueueMap::iterator q = __queues.find(SELF());
    assertion(q != __queues.end());
    __queues.unlock();
    unsigned usedAcc;

    if (acc != ACC_AUTO_BIND) {
        assertion(acc < int(__accs.size()));
        usedAcc = acc;
    }
    else {
        // Bind the new Context to the accelerator with less contexts
        // attached to it
        usedAcc = 0;
        for (unsigned i = 1; i < __accs.size(); i++) {
            if (__accs[i]->nContexts() < __accs[usedAcc]->nContexts()) {
                usedAcc = i;
            }
        }
    }
	unlock();

	trace("Created Execution Mode on Acc#%d", usedAcc);
    popState();

    // Initialize the global shared memory for the context
    Mode *mode = new Mode(__accs[usedAcc]);
    //manager->initShared(ctx);
    __modes.push_back(mode);

    return mode;
}

gmacError_t Process::migrate(Mode *mode, int acc)
{
	lockWrite();
    if (acc >= int(__accs.size())) return gmacErrorInvalidValue;
    gmacError_t ret = gmacSuccess;
	trace("Migrating execution mode");
    if (mode != NULL) {
#ifndef USE_MMAP
        if (int(mode->context().accId()) != acc) {
            // Create a new context in the requested accelerator
            ret = __accs[acc]->bind(ctx);
        }
#else
        fatal("Migration not implemented when using mmap");
#endif
    } else {
        // Create the context in the requested accelerator
        __accs[acc]->create();
    }
	trace("Context migrated");
	unlock();
    return ret;
}


void Process::remove(Mode *mode)
{
	__modes.remove(mode);
}

void Process::addAccelerator(Accelerator *acc)
{
	__accs.push_back(acc);
	__totalMemory += acc->memory();
}

void *Process::translate(void *addr)
{
    memory::Map &map = Mode::current()->map();
    memory::Object *object = map.find(addr);
    if(object == NULL) object = __shared.find(addr);
    if(object == NULL) return NULL;
    off_t offset = (uint8_t *)addr - (uint8_t *)object->addr();
    uint8_t *ret= (uint8_t *)object->device() + offset;
    return ret; 
}

void Process::send(THREAD_ID id)
{
    Mode *mode = Mode::current();
    __queues.lockRead();
    QueueMap::iterator q = __queues.find(id);
    assertion(q != __queues.end());
    __queues.unlock();
    q->second->lock();
    q->second->queue->push(mode);
    q->second->unlock();
    mode->inc();
    mode->detach();
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Mode::current()->detach();
    // Get a fresh context
    __queues.lockRead();
    QueueMap::iterator q = __queues.find(SELF());
    assertion(q != __queues.end());
    __queues.unlock();
    q->second->queue->pop()->attach();
}

void Process::sendReceive(THREAD_ID id)
{
    Mode * mode = Mode::current();
    __queues.lockRead();
	QueueMap::iterator q = __queues.find(id);
	assertion(q != __queues.end());
    __queues.unlock();
    q->second->lock();
	q->second->queue->push(mode);
    q->second->unlock();
    Context::key.set(NULL);
    __queues.lockRead();
	q = __queues.find(SELF());
	assertion(q != __queues.end());
    q->second->queue->pop()->attach();
    __queues.unlock();
}

void Process::copy(THREAD_ID id)
{
    Mode *mode = Mode::current();
    __queues.lockRead();
    QueueMap::iterator q = __queues.find(id);
    assertion(q != __queues.end());
    __queues.unlock();
    mode->inc();
    q->second->lock();
    q->second->queue->push(mode);
    q->second->unlock();
}

}
