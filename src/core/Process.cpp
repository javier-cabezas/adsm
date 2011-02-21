#include "Accelerator.h"
#include "IOBuffer.h"
#include "Mode.h"
#include "Process.h"

#include "allocator/Buddy.h"

#include "gmac/init.h"
#include "memory/Manager.h"
#include "memory/Object.h"
#include "memory/DistributedObject.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

ModeMap::ModeMap() :
    gmac::util::RWLock("ModeMap")
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
    gmac::util::RWLock("QueueMap")
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
QueueMap::insert(THREAD_T tid, ThreadQueue *q)
{
    lockWrite();
    std::pair<iterator, bool> ret =
        Parent::insert(value_type(tid, q));
    unlock();
    return ret;
}


void QueueMap::push(THREAD_T id, Mode &mode)
{
    lockRead();
    iterator i = Parent::find(id);
    if(i != Parent::end()) {
        i->second->queue->push(&mode);
    }
    unlock();
}

void QueueMap::attach()
{
    lockRead();
    iterator q = Parent::find(util::GetThreadId());
    if(q != Parent::end())
        q->second->queue->pop()->attach();
    unlock();
}

void QueueMap::erase(THREAD_T id)
{
    lockWrite();
    iterator i = Parent::find(id);
    if(i != Parent::end()) {
        delete i->second;
        Parent::erase(i);
    }
    unlock();
}

Process::Process() :
    util::Singleton<Process>(),
    gmac::util::RWLock("Process"),
    protocol_(*memory::ProtocolInit(GLOBAL_PROTOCOL)),
    shared_("SharedMemoryMap"),
    global_("GlobalMemoryMap"),
    orphans_("OrhpanMemoryMap"),
    current_(0)
{
    memory::Init();
    // Create the private per-thread variables for the implicit thread
    Mode::init();
    initThread();
}

Process::~Process()
{
    TRACE(LOCAL,"Cleaning process");
    while(modes_.empty() == false) {
        Mode *mode = modes_.begin()->first;
        mode->release(); 
    }

    std::vector<Accelerator *>::iterator a;
    for(a = accs_.begin(); a != accs_.end(); a++)
        delete *a;
    accs_.clear();
    queues_.cleanup();
    delete &protocol_;
    memory::Fini();
}

void Process::initThread()
{
    ThreadQueue * q = new core::ThreadQueue();
    queues_.insert(util::GetThreadId(), q);
    // Set the private per-thread variables
    Mode::initThread();
}

void Process::finiThread()
{
    queues_.erase(util::GetThreadId());
    Mode::finiThread();
}

Mode *Process::createMode(int acc)
{
    lockWrite();
    unsigned usedAcc;

    if (acc != ACC_AUTO_BIND) {
        ASSERTION(acc < int(accs_.size()));
        usedAcc = acc;
    }
    else {
        // Bind the new Context to the accelerator with less contexts
        // attached to it
        usedAcc = 0;
        for (unsigned i = 0; i < accs_.size(); i++) {
            if (accs_[i]->load() < accs_[usedAcc]->load()) {
                usedAcc = i;
            }
        }
    }

    TRACE(LOCAL,"Creatintg Execution Mode on Acc#%d", usedAcc);

    // Initialize the global shared memory for the context
    Mode *mode = accs_[usedAcc]->createMode(*this);
    modes_.insert(mode, accs_[usedAcc]);

    mode->attach();

    TRACE(LOCAL,"Adding "FMT_SIZE" global memory objects", global_.size());
    memory::Map::addOwner(*this, *mode);

    unlock();

    return mode;
}

void Process::removeMode(Mode &mode)
{
    lockWrite();
    TRACE(LOCAL, "Removing Execution Mode %p", &mode);
    modes_.remove(mode);
    unlock();
}


gmacError_t Process::globalMalloc(memory::Object &object)
{
    ModeMap::iterator i;
    lockRead();
    for(i = modes_.begin(); i != modes_.end(); i++) {
        if(object.addOwner(*i->first) != gmacSuccess) goto cleanup;
    }
    unlock();
    global_.insert(object);
    return gmacSuccess;
cleanup:
    ModeMap::iterator j;
    for(j = modes_.begin(); j != i; j++) {
        object.removeOwner(*j->first);
    }
    unlock();
    return gmacErrorMemoryAllocation;

}

gmacError_t Process::globalFree(memory::Object &object)
{
    if(global_.remove(object) == false) return gmacErrorInvalidValue;
    ModeMap::iterator i;
    lockRead();
    for(i = modes_.begin(); i != modes_.end(); i++) {
        object.removeOwner(*i->first);
    }
    unlock();
    return gmacSuccess;
}

// This function owns the global lock
gmacError_t Process::migrate(Mode &mode, int acc)
{
    if (acc >= int(accs_.size())) return gmacErrorInvalidValue;
    gmacError_t ret = gmacSuccess;
    TRACE(LOCAL,"Migrating execution mode");
#ifndef USE_MMAP
    if (int(mode.getAccelerator().id()) != acc) {
        // Create a new context in the requested accelerator
        //ret = _accs[acc]->bind(mode);
        ret = mode.moveTo(*accs_[acc]);

        if (ret == gmacSuccess) {
            modes_[&mode] = accs_[acc];
        }
    }
#else
    FATAL("Migration not implemented when using mmap");
#endif
    TRACE(LOCAL,"Context migrated");
    return ret;
}

void Process::addAccelerator(Accelerator &acc)
{
    accs_.push_back(&acc);
}


accptr_t Process::translate(const hostptr_t addr)
{
    Mode &mode = Mode::getCurrent();
    memory::Object *object = mode.getObject(addr);
    if(object == NULL) return accptr_t(NULL);
    accptr_t ptr = object->acceleratorAddr(addr);
    object->release();
    return ptr;
}

void Process::send(THREAD_T id)
{
    Mode &mode = Mode::getCurrent();
    queues_.push(id, mode);
    mode.use();
    mode.detach();
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Mode::getCurrent().detach();
    // Get a fresh context
    queues_.attach();
}

void Process::sendReceive(THREAD_T id)
{
    Mode &mode = Mode::getCurrent();
    queues_.push(id, mode);
    Mode::initThread();
    queues_.attach();
}

void Process::copy(THREAD_T id)
{
    Mode &mode = Mode::getCurrent();
    queues_.push(id, mode);
    mode.use();
}

Mode *Process::owner(const hostptr_t addr, size_t size) const
{
    // We consider global objects for ownership,
    // since it contains distributed objects not found in shared_
    memory::Object *object = shared_.get(addr, size);
    if(object == NULL) object = global_.get(addr, size);
    if(object == NULL) return NULL;
    Mode &ret = object->owner(addr);
    object->release();
    return &ret;
}

bool Process::allIntegrated()
{
    bool ret = true;
    std::vector<Accelerator *>::iterator a;
    for(a = accs_.begin(); a != accs_.end(); a++) {
        ret = ret && (*a)->integrated();
    }
    return ret;
}

}}
