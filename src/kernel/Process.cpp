#include "Process.h"
#include "Mode.h"
#include "Accelerator.h"
#include "allocator/Buddy.h"

#include <gmac/init.h>
#include <memory/Manager.h>
#include <memory/Object.h>
#include <memory/DistributedObject.h>
#include <trace/Thread.h>


namespace gmac {

Process *proc = NULL;

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

size_t ModeMap::remove(Mode &mode)
{
    lockWrite();
    size_type ret = Parent::erase(&mode);
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

    if(_ioMemory != NULL) delete _ioMemory;
    _ioMemory = NULL;

    std::vector<Accelerator *>::iterator a;
    std::list<Mode *>::iterator c;
    _modes.lockWrite();
    ModeMap::const_iterator i;
    for(i = _modes.begin(); i != _modes.end(); i++) {
        i->first->release();
        delete i->first;
    }
    _modes.clear();
    _modes.unlock();

    for(a = _accs.begin(); a != _accs.end(); a++)
        delete *a;
    _accs.clear();
    // TODO: Free buddy allocator
    _queues.cleanup();
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
    // Create the private per-thread variables
    // Register first, implicit, thread
    Mode::init();

    proc->initThread();
}

void
Process::initThread()
{
    ThreadQueue * q = new ThreadQueue();
    _queues.insert(SELF(), q);
    // Set the private per-thread variables
    Mode::initThread();
}

Mode *Process::create(int acc)
{
    lockWrite();
    unsigned usedAcc;

    if (acc != ACC_AUTO_BIND) {
        assertion(acc < int(_accs.size()));
        usedAcc = acc;
    }
    else {
        // Bind the new Context to the accelerator with less contexts
        // attached to it
        usedAcc = 0;
        for (unsigned i = 0; i < _accs.size(); i++) {
            if (_accs[i]->load() < _accs[usedAcc]->load()) {
                usedAcc = i;
            }
        }
    }

	trace("Creatintg Execution Mode on Acc#%d", usedAcc);

    // Initialize the global shared memory for the context
    Mode *mode = _accs[usedAcc]->createMode();
    _accs[usedAcc]->registerMode(*mode);
    _modes.insert(mode, _accs[usedAcc]);

    trace("Adding %zd shared memory objects", __shared.size());
    memory::Map::iterator i;
    for(i = __shared.begin(); i != __shared.end(); i++) {
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->addOwner(*mode);
    }
	unlock();

    mode->attach();
    lockWrite();
    if(_ioMemory == NULL)
        _ioMemory = new kernel::allocator::Buddy(_accs.size() * paramIOMemory);
    unlock();
    return mode;
}

#ifndef USE_MMAP
gmacError_t Process::globalMalloc(memory::DistributedObject &object, size_t size)
{
    gmacError_t ret;
    ModeMap::iterator i;
    lockRead();
    for(i = _modes.begin(); i != _modes.end(); i++) {
        if((ret = object.addOwner(*i->first)) != gmacSuccess) goto cleanup;
    }
    unlock();
    return gmacSuccess;
cleanup:
    ModeMap::iterator j;
    for(j = _modes.begin(); j != i; j++) {
        object.removeOwner(*j->first);
    }
    unlock();
    return gmacErrorMemoryAllocation;

}

gmacError_t Process::globalFree(memory::DistributedObject &object)
{
    gmacError_t ret = gmacSuccess;
    ModeMap::iterator i;
    lockRead();
    for(i = _modes.begin(); i != _modes.end(); i++) {
        gmacError_t tmp = object.removeOwner(*i->first);
        if(tmp != gmacSuccess) ret = tmp;
    }
    unlock();
    return gmacSuccess;
}
#endif

gmacError_t Process::migrate(Mode &mode, int acc)
{
	lockWrite();
    if (acc >= int(_accs.size())) return gmacErrorInvalidValue;
    gmacError_t ret = gmacSuccess;
	trace("Migrating execution mode");
#ifndef USE_MMAP
    if (int(mode.accId()) != acc) {
        // Create a new context in the requested accelerator
        //ret = _accs[acc]->bind(mode);
        ret = mode.moveTo(*_accs[acc]);

        if (ret == gmacSuccess) {
            _modes[&mode] = _accs[acc];                 
        }
    }
#else
    Fatal("Migration not implemented when using mmap");
#endif
    unlock();
	trace("Context migrated");
    return ret;
    return gmacErrorUnknown;
}


void Process::remove(Mode &mode)
{
    trace("Adding %zd shared memory objects", __shared.size());
    memory::Map::iterator i;
    for(i = __shared.begin(); i != __shared.end(); i++) {
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->removeOwner(mode);
    }
    lockWrite();
	_modes.remove(mode);
    unlock();   
}

void Process::addAccelerator(Accelerator *acc)
{
	_accs.push_back(acc);
}

IOBuffer *Process::createIOBuffer(size_t size)
{
    assertion(_ioMemory != NULL);
    void *addr = _ioMemory->get(size);
    if(addr == NULL) return NULL;
    return new IOBuffer(addr, size);
}

void Process::destroyIOBuffer(IOBuffer *buffer)
{
    if(_ioMemory == NULL) return;
    _ioMemory->put(buffer->addr(), buffer->size());
    delete buffer;
}


void *Process::translate(void *addr)
{
    Mode &mode = Mode::current();
    const memory::Object *object = mode.getObjectRead(addr);
    if(object == NULL) return NULL;
    void * ptr = object->device(addr);
    mode.putObject(*object);

    return ptr;
}

void Process::send(THREAD_ID id)
{
    Mode &mode = Mode::current();
    QueueMap::iterator q = _queues.find(id);
    assertion(q != _queues.end());
    q->second->queue->push(&mode);
    mode.inc();
    mode.detach();
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Mode::current().detach();
    // Get a fresh context
    QueueMap::iterator q = _queues.find(SELF());
    assertion(q != _queues.end());
    q->second->queue->pop()->attach();
}

void Process::sendReceive(THREAD_ID id)
{
    Mode &mode = Mode::current();
	QueueMap::iterator q = _queues.find(id);
	assertion(q != _queues.end());
	q->second->queue->push(&mode);
    Mode::initThread();
	q = _queues.find(SELF());
	assertion(q != _queues.end());
    q->second->queue->pop()->attach();
}

void Process::copy(THREAD_ID id)
{
    Mode &mode = Mode::current();
    QueueMap::iterator q = _queues.find(id);
    assertion(q != _queues.end());
    mode.inc();
    q->second->queue->push(&mode);
}

Mode *Process::owner(const void *addr) const
{
    const memory::Object *object = __global.getObjectRead(addr);
    if(object == NULL) return NULL;
    Mode & ret = object->owner();
    __global.putObject(*object);
    return &ret;
}


}
