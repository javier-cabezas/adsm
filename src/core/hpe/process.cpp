#include <sstream>

#include "core/io_buffer.h"

//#include "core/hpe/Accelerator.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/process.h"
#include "core/hpe/thread.h"

#include "memory/Manager.h"
#include "memory/object.h"
#include "trace/Tracer.h"

namespace __impl { namespace core { namespace hpe {

vdeviceMap::vdeviceMap() :
    gmac::util::Lock("vdeviceMap")
{}

std::pair<vdeviceMap::iterator, bool>
vdeviceMap::insert(vdevice *mode)
{
    lock();
    std::pair<iterator, bool> ret = Parent::insert(value_type(mode, 1));
    if(ret.second == false) ret.first->second++;
    unlock();
    return ret;
}

void vdeviceMap::remove(vdevice &mode)
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


void QueueMap::push(THREAD_T id, vdevice &mode)
{
    lockRead();
    iterator i = Parent::find(id);
    if(i != Parent::end()) {
        i->second->queue->push(&mode);
    }
    unlock();
}

vdevice *QueueMap::pop()
{
    vdevice *ret = NULL;
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

process::process() :
    core::process(),
    gmac::util::RWLock("process"),
    protocol_(*memory::ProtocolInit(GLOBAL_PROTOCOL)),
    shared_("SharedMemoryMap"),
    global_("GlobalMemoryMap"),
    orphans_("OrhpanMemoryMap"),
    current_(0),
    resourceManager_(*this)
{
    TLS::Init();
}

process::~process()
{
    finiThread();

#if 0
    while(modes_.empty() == false) {
        vdevice *mode = modes_.begin()->first;
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

    queues_.cleanup();
#endif
    delete &protocol_;
}

void process::init()
{
    // Create the private per-thread variables for the implicit thread
}

void process::initThread(bool userThread, THREAD_T tidParent)
{
#if 0
    ThreadQueue * q = new ThreadQueue();
    queues_.insert(util::GetThreadId(), q);
#endif

    if (userThread) {
        thread *parent = NULL;
        map_thread::iterator it = mapThreads_.find(tidParent);

        if (it != mapThreads_.end()) {
            parent = it->second;
        }

        // Set the private per-thread variables
        thread *t = new thread(*this);
        ASSERTION(mapThreads_.find(util::GetThreadId()) == mapThreads_.end(),
                "Thread already registered");

        mapThreads_.insert(map_thread::value_type(util::GetThreadId(), t));

        gmacError_t err = resourceManager_.init_thread(*t, parent);
        ASSERTION(err == gmacSuccess);
    }
}

void process::finiThread()
{
#if 0
    queues_.erase(util::GetThreadId());
#endif
    // TODO: remove vdevices in virtual device table
    //if (Thread::hasCurrentVirtualDevice() != false) removevdevice(Thread::getCurrentVirtualDevice());

    map_thread::iterator it = mapThreads_.find(util::GetThreadId());
    ASSERTION(it != mapThreads_.end(), "Thread not registered");
    mapThreads_.erase(it);

    delete &TLS::get_current_thread();
}

#if 0
vdevice *process::createvdevice(int acc)
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

    TRACE(LOCAL,"Creatintg Execution vdevice on Acc#%d", usedAcc);

    // Initialize the global shared memory for the context
    vdevice *mode = dynamic_cast<vdevice *>(accs_[usedAcc]->createvdevice(*this, *new AddressSpace("", *this, *accs_[usedAcc])));
    modes_.insert(mode);
    load_[usedAcc]++;
    unlock();

    TRACE(LOCAL,"Adding "FMT_SIZE" global memory objects", global_.size());
    AddressSpace::addOwner(*this, *mode);

    return mode;
}

void process::removevdevice(vdevice &mode)
{
    lockWrite();
    TRACE(LOCAL, "Removing Execution vdevice %p", &mode);
    load_[mode.getAccelerator().id()]--;
    modes_.remove(mode);
    AddressSpace::removeOwner(*this, mode);
    mode.decRef();
    unlock();
}

gmacError_t process::globalMalloc(memory::object &object)
{
    vdeviceMap::iterator i;
    lockRead();
    for(i = modes_.begin(); i != modes_.end(); i++) {
        if(object.addOwner(*i->first) != gmacSuccess) goto cleanup;
    }
    unlock();
    global_.addObject(object);
    return gmacSuccess;

cleanup:
    vdeviceMap::iterator j;
    for(j = modes_.begin(); j != i; j++) {
        object.removeOwner(*j->first);
    }
    unlock();
    return gmacErrorMemoryAllocation;

}

gmacError_t process::globalFree(memory::object &object)
{
    if(global_.removeObject(object) == false) return gmacErrorInvalidValue;
    vdeviceMap::iterator i;
    lockRead();
    for(i = modes_.begin(); i != modes_.end(); i++) {
        object.removeOwner(*i->first);
    }
    unlock();
    return gmacSuccess;
}
#endif

#if 0
// This function owns the global lock
gmacError_t process::migrate(int acc)
{
    if (acc >= int(accs_.size())) return gmacErrorInvalidValue;
    if(Thread::hasCurrentVirtualDevice() == false) {
        vdevice *mode = createvdevice(acc);
        Thread::setCurrentVirtualDevice(*mode);
        return gmacSuccess;
    }
    vdevice &mode = Thread::getCurrentVirtualDevice();
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

void process::addAccelerator(Accelerator &acc)
{
    accs_.push_back(&acc);
    load_.push_back(0);
    std::stringstream ss;
    ss << "Accelerator " << acc.id();
    //aSpaces_.push_back(new AddressSpace(ss.str().c_str(), *this));;
}


accptr_t process::translate(const hostptr_t addr)
{
    vdevice &dev = Thread::getCurrentVirtualDevice();
    memory::map_object &map = dev.get_aspace();
    memory::object *object = map.getObject(addr);
    if(object == NULL) return accptr_t(0);
    accptr_t ptr = object->acceleratorAddr(mode, addr);
    object->decRef();
    return ptr;
}

void process::send(THREAD_T id)
{
    if (Thread::hasCurrentVirtualDevice() == false) return;
    vdevice &mode = Thread::getCurrentVirtualDevice();
    mode.wait();
    queues_.push(id, mode);
    Thread::setCurrentVirtualDevice(NULL);
}

void process::receive()
{
    // Get current context and destroy (if necessary)
    if(Thread::hasCurrentVirtualDevice()) Thread::getCurrentVirtualDevice().decRef();
    // Get a fresh context
    Thread::setCurrentVirtualDevice(queues_.pop());
}

void process::sendReceive(THREAD_T id)
{
    if(Thread::hasCurrentVirtualDevice()) {
        Thread::getCurrentVirtualDevice().wait();
        queues_.push(id, Thread::getCurrentVirtualDevice());
    }
    Thread::setCurrentVirtualDevice(queues_.pop());
}

void process::copy(THREAD_T id)
{
    if(Thread::hasCurrentVirtualDevice() == false) return;
    vdevice &mode = Thread::getCurrentVirtualDevice();
    queues_.push(id, mode);
    mode.incRef();
    modes_.insert(&mode);
}

core::address_space *process::owner(const hostptr_t addr, size_t size)
{
    // We consider global objects for ownership,
    // since it contains distributed objects not found in shared_
    memory::object *object = shared_.getObject(addr, size);
    if(object == NULL) object = global_.getObject(addr, size);
    if(object == NULL) return NULL;
    core::address_space &ret = object->owner();
    object->decRef();
    return &ret;
}

bool process::allIntegrated()
{
    bool ret = true;
    std::vector<AcceleratorPtr>::iterator a;
    for(a = accs_.begin(); a != accs_.end(); a++) {
        ret = ret && (*a)->hasIntegratedMemory();
    }
    return ret;
}

gmacError_t
process::prepareForCall()
{
    gmacError_t ret = gmacSuccess;
    vdeviceMap::iterator i;
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
#endif

}}}
