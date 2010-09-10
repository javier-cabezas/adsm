#include "Process.h"
#include "Mode.h"
#include "Accelerator.h"
#include "allocator/Buddy.h"

#include <gmac/init.h>
#include <memory/Manager.h>
#include <memory/Object.h>
#include <memory/DistributedObject.h>
#include <trace/Thread.h>


gmac::Process *proc = NULL;

namespace gmac {

ModeMap::ModeMap() :
    RWLock("ModeMap")
{}

std::pair<ModeMap::iterator, bool>
ModeMap::insert(Mode *mode, Accelerator *acc)
{
    lockWrite();
    std::pair<iterator, bool> ret = Parent::insert(value_type(mode, acc));
    unlock();
    return ret;
}

size_t ModeMap::remove(Mode *mode)
{
    lockWrite();
    size_type ret = Parent::erase(mode);
    unlock();
    return ret;
}

QueueMap::QueueMap() : 
    util::RWLock("QueueMap")
{}

void QueueMap::cleanup()
{
    QueueMap::iterator q;
    lockWrite();
    for(q = Parent::begin(); q != Parent::end(); q++)
        delete q->second;
    clear();
    unlock();
}

std::pair<QueueMap::iterator, bool>
QueueMap::insert(THREAD_ID tid, ThreadQueue *q)
{
    lockWrite();
    std::pair<iterator, bool> ret =
        Parent::insert(value_type(tid, q));
    unlock();
    return ret;
}

QueueMap::iterator QueueMap::find(THREAD_ID id)
{
    lockRead();
    iterator q = Parent::find(id);
    unlock();
    return q;
}

QueueMap::iterator QueueMap::end()
{
    lockRead();
    iterator ret = Parent::end();
    unlock();
    return ret;
}


size_t Process::__totalMemory = 0;

Process::Process() :
    RWLock("Process"),
    __global("GlobalMemoryMap"),
    __shared("SharedMemoryMap"),
    current(0),
    _ioMemory(NULL)
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
    if(_ioMemory != NULL) delete _ioMemory;
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
        for (unsigned i = 0; i < __accs.size(); i++) {
            if (__accs[i]->load() < __accs[usedAcc]->load()) {
                usedAcc = i;
            }
        }
    }

	trace("Creatintg Execution Mode on Acc#%d", usedAcc);

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

IOBuffer *Process::createIOBuffer(size_t size)
{
    if(_ioMemory == NULL) _ioMemory = new kernel::allocator::Buddy(paramIOMemory);
    return NULL;
}

void Process::destroyIOBuffer(IOBuffer *buffer)
{
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
