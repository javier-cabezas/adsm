#include "core/IOBuffer.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"

#include "memory/Manager.h"
#include "memory/Object.h"
#include "memory/DistributedObject.h"
#include "trace/Tracer.h"

namespace __impl { namespace core { namespace hpe {

ModeMap::ModeMap() :
    gmac::util::Lock("ModeMap")
{}

std::pair<ModeMap::iterator, bool>
ModeMap::insert(Mode *mode)
{
    lock();
    std::pair<iterator, bool> ret = Parent::insert(value_type(mode, 1));
    if(ret.second == false) ret.first->second++;
    unlock();
    return ret;
}

void ModeMap::remove(Mode &mode)
{
    lock();
    iterator i = Parent::find(&mode);
    if(i != Parent::end()) {
        i->second--;
        if(i->second == 0) Parent::erase(i);
    }
    unlock();
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

Mode *QueueMap::pop()
{
        Mode *ret = NULL;
    lockRead();
    iterator q = Parent::find(util::GetThreadId());
    if(q != Parent::end())
        ret = q->second->queue->pop();
    unlock();
        return ret;
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
    core::Process(),
    gmac::util::RWLock("Process"),
    protocol_(*memory::ProtocolInit(GLOBAL_PROTOCOL)),
    shared_("SharedMemoryMap"),
    global_("GlobalMemoryMap"),
    orphans_("OrhpanMemoryMap"),
    current_(0)
{
    // Create the private per-thread variables for the implicit thread
    util::Private<Mode>::init(CurrentMode_);
    initThread();
}

Process::~Process()
{
    while(modes_.empty() == false) {
        Mode *mode = modes_.begin()->first;
        ASSERTION(mode != NULL);
        modes_.remove(*mode);
        mode->decRef();
    }

    std::vector<Accelerator *>::iterator a;
    for(a = accs_.begin(); a != accs_.end(); a++)
        delete *a;
    accs_.clear();
    queues_.cleanup();
    delete &protocol_;
}

void Process::initThread()
{
    ThreadQueue * q = new ThreadQueue();
    queues_.insert(util::GetThreadId(), q);
    // Set the private per-thread variables
    CurrentMode_.set(NULL);
}

void Process::finiThread()
{
    queues_.erase(util::GetThreadId());
    Mode *mode = CurrentMode_.get();
    if(mode != NULL) removeMode(*mode);
    CurrentMode_.set(NULL);
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
    Mode *mode = dynamic_cast<Mode *>(accs_[usedAcc]->createMode(*this));
    modes_.insert(mode);
    unlock();

    TRACE(LOCAL,"Adding "FMT_SIZE" global memory objects", global_.size());
    Map::addOwner(*this, *mode);

    return mode;
}

void Process::removeMode(Mode &mode)
{
    lockWrite();
    TRACE(LOCAL, "Removing Execution Mode %p", &mode);
    modes_.remove(mode);
    mode.decRef();
    unlock();
}

Mode &Process::getCurrentMode()
{
    Mode *mode = CurrentMode_.get();
    if(mode != NULL) return *mode;
    mode = createMode();
    CurrentMode_.set(mode);
    return *mode;
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
gmacError_t Process::migrate(int acc)
{
    if (acc >= int(accs_.size())) return gmacErrorInvalidValue;
    Mode *mode = CurrentMode_.get();
    if(mode == NULL) {
        mode = createMode(acc);
        CurrentMode_.set(mode);
        return gmacSuccess;
    }
    gmacError_t ret = gmacSuccess;
    TRACE(LOCAL,"Migrating execution mode");
#ifndef USE_MMAP
    if (int(mode->getAccelerator().id()) != acc) {
        // Create a new context in the requested accelerator
        //ret = _accs[acc]->bind(mode);
        ret = mode->moveTo(*accs_[acc]);
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
    Mode *mode = CurrentMode_.get();
    if(mode == NULL) return accptr_t(0);
    memory::Object *object = mode->getObject(addr);
    if(object == NULL) return accptr_t(0);
    accptr_t ptr = object->acceleratorAddr(*mode, addr);
    object->decRef();
    return ptr;
}

void Process::send(THREAD_T id)
{
    Mode *mode = CurrentMode_.get();
        if(mode == NULL) return;
    mode->wait();
    queues_.push(id, *mode);
        CurrentMode_.set(NULL);
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    Mode *mode = CurrentMode_.get();
    if(mode != NULL) mode->decRef();
    // Get a fresh context
    CurrentMode_.set(queues_.pop());
}

void Process::sendReceive(THREAD_T id)
{
    Mode *mode = CurrentMode_.get();
    if(mode != NULL) {
        mode->wait();
        queues_.push(id, *mode);
    }
    CurrentMode_.set(queues_.pop());
}

void Process::copy(THREAD_T id)
{
    Mode *mode = CurrentMode_.get();
    if(mode == NULL) return;
    queues_.push(id, *mode);
    mode->incRef();
    modes_.insert(mode);
}

core::Mode *Process::owner(const hostptr_t addr, size_t size)
{
    // We consider global objects for ownership,
    // since it contains distributed objects not found in shared_
    memory::Object *object = shared_.get(addr, size);
    if(object == NULL) object = global_.get(addr, size);
    if(object == NULL) return NULL;
    core::Mode &ret = object->owner(getCurrentMode(), addr);
    object->decRef();
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

gmacError_t
Process::prepareForCall()
{
    gmacError_t ret = gmacSuccess;
    ModeMap::iterator i;
    lockRead();
    if (global_.size() > 0) {
        for(i = modes_.begin(); i != modes_.end(); i++) {
            ret = i->first->prepareForCall();
            if (ret != gmacSuccess) break;
        }
    }
    unlock();

    return ret;
}


}}}
