#include <sstream>

#include "core/IOBuffer.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Mode.h"
#include "core/hpe/Process.h"
#include "core/hpe/Thread.h"

#include "memory/Manager.h"
#include "memory/Object.h"
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

#if 0
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
#endif

Process::Process() :
    core::Process(),
    gmac::util::RWLock("Process"),
    protocol_(*memory::ProtocolInit(GLOBAL_PROTOCOL)),
    shared_("SharedMemoryMap", *this),
    global_("GlobalMemoryMap", *this),
    orphans_("OrhpanMemoryMap", *this),
    current_(0),
    resourceManager_(*this)
{
    TLS::Init();

    // Create the private per-thread variables for the implicit thread
    initThread();
}

Process::~Process()
{
    finiThread();

    while(modes_.empty() == false) {
        Mode *mode = modes_.begin()->first;
        ASSERTION(mode != NULL);
        modes_.remove(*mode);
        mode->decRef();
    }

    std::vector<Accelerator *>::iterator it;
    for (it = accs_.begin(); it != accs_.end(); ++it) {
        Accelerator *acc = *it;
        ASSERTION(acc != NULL);
        delete acc;
    }

#if 0
    queues_.cleanup();
#endif
    delete &protocol_;
}

void Process::initThread()
{
#if 0
    ThreadQueue * q = new ThreadQueue();
    queues_.insert(util::GetThreadId(), q);
#endif

    // Set the private per-thread variables
    new Thread(*this);
}

void Process::finiThread()
{
#if 0
    queues_.erase(util::GetThreadId());
#endif
    if (Thread::hasCurrentVirtualDevice() != false) removeMode(Thread::getCurrentVirtualDevice());
    delete &TLS::getCurrentThread();
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
        for (unsigned i = 1; i < accs_.size(); i++) {
            if (load_[i] < load_[usedAcc]) {
                usedAcc = i;
            }
        }
    }

    TRACE(LOCAL,"Creatintg Execution Mode on Acc#%d", usedAcc);

    // Initialize the global shared memory for the context
    Mode *mode = dynamic_cast<Mode *>(accs_[usedAcc]->createMode(*this, *new AddressSpace("", *this, *accs_[usedAcc])));
    modes_.insert(mode);
    load_[usedAcc]++;
    unlock();

    TRACE(LOCAL,"Adding "FMT_SIZE" global memory objects", global_.size());
    AddressSpace::addOwner(*this, *mode);

    return mode;
}

void Process::removeMode(Mode &mode)
{
    lockWrite();
    TRACE(LOCAL, "Removing Execution Mode %p", &mode);
    load_[mode.getAccelerator().id()]--;
    modes_.remove(mode);
    AddressSpace::removeOwner(*this, mode);
    mode.decRef();
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
    global_.addObject(object);
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
    if(global_.removeObject(object) == false) return gmacErrorInvalidValue;
    ModeMap::iterator i;
    lockRead();
    for(i = modes_.begin(); i != modes_.end(); i++) {
        object.removeOwner(*i->first);
    }
    unlock();
    return gmacSuccess;
}

#if 0
// This function owns the global lock
gmacError_t Process::migrate(int acc)
{
    if (acc >= int(accs_.size())) return gmacErrorInvalidValue;
    if(Thread::hasCurrentVirtualDevice() == false) {
        Mode *mode = createMode(acc);
        Thread::setCurrentVirtualDevice(*mode);
        return gmacSuccess;
    }
    Mode &mode = Thread::getCurrentVirtualDevice();
    gmacError_t ret = gmacSuccess;
    TRACE(LOCAL,"Migrating execution mode");
#ifndef USE_MMAP
    if (int(mode.getAccelerator().id()) != acc) {
        // Create a new context in the requested accelerator
        //ret = _accs[acc]->bind(mode);
        ret = mode.moveTo(*accs_[acc]);
    }
#else
    FATAL("Migration not implemented when using mmap");
#endif
    TRACE(LOCAL,"Context migrated");
    return ret;
}
#endif

void Process::addAccelerator(Accelerator &acc)
{
    accs_.push_back(&acc);
    load_.push_back(0);
    std::stringstream ss;
    ss << "Accelerator " << acc.id();
    //aSpaces_.push_back(new AddressSpace(ss.str().c_str(), *this));;
}

accptr_t Process::translate(const hostptr_t addr)
{
    Mode &mode = Thread::getCurrentVirtualDevice();
    memory::ObjectMap &map = mode.getAddressSpace();
    memory::Object *object = map.getObject(addr);
    if(object == NULL) return accptr_t(0);
    accptr_t ptr = object->acceleratorAddr(mode, addr);
    object->decRef();
    return ptr;
}

#if 0
void Process::send(THREAD_T id)
{
    if (Thread::hasCurrentVirtualDevice() == false) return;
    Mode &mode = Thread::getCurrentVirtualDevice();
    mode.wait();
    queues_.push(id, mode);
    Thread::setCurrentVirtualDevice(NULL);
}

void Process::receive()
{
    // Get current context and destroy (if necessary)
    if(Thread::hasCurrentVirtualDevice()) Thread::getCurrentVirtualDevice().decRef();
    // Get a fresh context
    Thread::setCurrentVirtualDevice(queues_.pop());
}

void Process::sendReceive(THREAD_T id)
{
    if(Thread::hasCurrentVirtualDevice()) {
        Thread::getCurrentVirtualDevice().wait();
        queues_.push(id, Thread::getCurrentVirtualDevice());
    }
    Thread::setCurrentVirtualDevice(queues_.pop());
}

void Process::copy(THREAD_T id)
{
    if(Thread::hasCurrentVirtualDevice() == false) return;
    Mode &mode = Thread::getCurrentVirtualDevice();
    queues_.push(id, mode);
    mode.incRef();
    modes_.insert(&mode);
}
#endif

core::Mode *Process::owner(const hostptr_t addr, size_t size)
{
    // We consider global objects for ownership,
    // since it contains distributed objects not found in shared_
    memory::Object *object = shared_.getObject(addr, size);
    if(object == NULL) object = global_.getObject(addr, size);
    if(object == NULL) return NULL;
    core::Mode &ret = object->owner(Thread::getCurrentVirtualDevice(), addr);
    object->decRef();
    return &ret;
}

bool Process::allIntegrated()
{
    bool ret = true;
    std::vector<AcceleratorPtr>::iterator a;
    for(a = accs_.begin(); a != accs_.end(); a++) {
        ret = ret && (*a)->hasIntegratedMemory();
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
