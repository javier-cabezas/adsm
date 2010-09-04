#include "Process.h"
#include "Mode.h"
#include "Accelerator.h"

#include <memory/Manager.h>
#include <memory/Object.h>
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

ModeMap::ModeMap() :
    RWLock(LockModeMap)
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
    __modes.lockWrite();
    ModeMap::const_iterator i;
    for(i = __modes.begin(); i != __modes.end(); i++) {
        delete i->first;
    }
    __modes.clear();
    __modes.unlock();

    for(a = __accs.begin(); a != __accs.end(); a++)
        delete *a;
    __accs.clear();
    __queues.cleanup();
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
    Mode::init();
    proc->initThread();
}

void
Process::initThread()
{
    ThreadQueue * q = new ThreadQueue();
    __queues.insert(SELF(), q);
}

Mode *Process::create(int acc)
{
    pushState(Init);
    lockWrite();
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
            if (__accs[i]->load() < __accs[usedAcc]->load()) {
                usedAcc = i;
            }
        }
    }

	trace("Creatintg Execution Mode on Acc#%d", usedAcc);
    popState();

    // Initialize the global shared memory for the context
    Mode *mode = __accs[usedAcc]->createMode();
    __modes.insert(mode, __accs[usedAcc]);

    trace("Adding %zd shared memory objects", __shared.size());
    memory::Map::iterator i;
    for(i = __shared.begin(); i != __shared.end(); i++) {
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->addOwner(mode);
    }
	unlock();

    mode->attach();
    return mode;
}

#ifndef USE_MMAP
gmacError_t Process::globalMalloc(memory::DistributedObject &object, size_t size)
{
    gmacError_t ret;
    ModeMap::iterator i;
    lockRead();
    for(i = __modes.begin(); i != __modes.end(); i++) {
        if((ret = object.addOwner(i->first)) != gmacSuccess) goto cleanup;
    }
    unlock();
    return gmacSuccess;
cleanup:
    ModeMap::iterator j;
    for(j = __modes.begin(); j != i; j++) {
        object.removeOwner(j->first);
    }
    unlock();
    return gmacErrorMemoryAllocation;

}

gmacError_t Process::globalFree(memory::DistributedObject &object)
{
    gmacError_t ret = gmacSuccess;
    ModeMap::iterator i;
    lockRead();
    for(i = __modes.begin(); i != __modes.end(); i++) {
        gmacError_t tmp = object.removeOwner(i->first);
        if(tmp != gmacSuccess) ret = tmp;
    }
    unlock();
    return gmacSuccess;
}
#endif

gmacError_t Process::migrate(Mode *mode, int acc)
{
#if 0
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
#endif
    fatal("Not supported yet!");
    return gmacErrorUnknown;
}


void Process::remove(Mode *mode)
{
    trace("Adding %zd shared memory objects", __shared.size());
    memory::Map::iterator i;
    for(i = __shared.begin(); i != __shared.end(); i++) {
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->removeOwner(mode);
    }
    lockWrite();
	__modes.remove(mode);
    unlock();   
}

void Process::addAccelerator(Accelerator *acc)
{
	__accs.push_back(acc);
}

void *Process::translate(void *addr)
{
    memory::Object *object = Mode::current()->findObject(addr);
    if(object == NULL) object = __shared.find(addr);
    if(object == NULL) return NULL;
    return object->device(addr); 
}

void Process::send(THREAD_ID id)
{
    Mode *mode = Mode::current();
    QueueMap::iterator q = __queues.find(id);
    assertion(q != __queues.end());
    q->second->queue->push(mode);
    mode->inc();
    mode->detach();
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Mode::current()->detach();
    // Get a fresh context
    QueueMap::iterator q = __queues.find(SELF());
    assertion(q != __queues.end());
    q->second->queue->pop()->attach();
}

void Process::sendReceive(THREAD_ID id)
{
    Mode * mode = Mode::current();
	QueueMap::iterator q = __queues.find(id);
	assertion(q != __queues.end());
	q->second->queue->push(mode);
    Mode::init();
	q = __queues.find(SELF());
	assertion(q != __queues.end());
    q->second->queue->pop()->attach();
}

void Process::copy(THREAD_ID id)
{
    Mode *mode = Mode::current();
    QueueMap::iterator q = __queues.find(id);
    assertion(q != __queues.end());
    mode->inc();
    q->second->queue->push(mode);
}

Mode *Process::owner(const void *addr)
{
    memory::Object *object = __global.find(addr);
    if(object == NULL) return NULL;
    return object->owner();
}


}
