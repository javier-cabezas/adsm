#include <sstream>

//#include "core/hpe/Accelerator.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/process.h"
#include "core/hpe/thread.h"

#include "memory/Manager.h"
#include "memory/object.h"
#include "trace/Tracer.h"

namespace __impl { namespace core { namespace hpe {

vdeviceMap::vdeviceMap() :
    gmac::util::mutex("vdeviceMap")
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

process::process() :
    core::process(),
    gmac::util::lock_rw("process"),
    current_(0),
    resourceManager_(*this)
{
    TLS::Init();
}

process::~process()
{
    if (thread::has_current_thread()) finiThread();
}

void process::init()
{
    // Create the private per-thread variables for the implicit thread
}

void process::initThread(bool userThread, THREAD_T tidParent)
{
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
    // TODO: remove vdevices in virtual device table
    //if (Thread::hasCurrentVirtualDevice() != false) removevdevice(Thread::getCurrentVirtualDevice());

    map_thread::iterator it = mapThreads_.find(util::GetThreadId());
    ASSERTION(it != mapThreads_.end(), "Thread not registered");
    mapThreads_.erase(it);

    delete &TLS::get_current_thread();
}

}}}
