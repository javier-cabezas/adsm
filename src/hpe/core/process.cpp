#include <sstream>

#include "hpe/core/vdevice.h"
#include "hpe/core/process.h"
#include "hpe/core/thread.h"

#include "trace/Tracer.h"

namespace __impl { namespace core { namespace hpe {

map_vdevice::map_vdevice() :
    lock("map_vdevice")
{}

std::pair<map_vdevice::iterator, bool>
map_vdevice::insert(vdevice *mode)
{
    lock::lock();
    std::pair<iterator, bool> ret = Parent::insert(value_type(mode, 1));
    if(ret.second == false) ret.first->second++;
    lock::unlock();
    return ret;
}

void map_vdevice::remove(vdevice &mode)
{
    lock::lock();
    iterator i = Parent::find(&mode);
    if(i != Parent::end()) {
        i->second--;
        if(i->second == 0) Parent::erase(i);
    }
    lock::unlock();
}

process::process() :
    lock("process"),
    current_(0),
    resourceManager_(*this),
    mapThreads_("map_thread")
{
    TLS::Init();
}

process::~process()
{
#if 0
    if (thread::has_current_thread()) fini_thread(true);
#endif
}

void process::init()
{
    // Create the private per-thread variables for the implicit thread
}

void process::init_thread(bool userThread, THREAD_T tidParent)
{
    if (userThread) {
        thread *parent = NULL;
        map_thread::iterator it = mapThreads_.find(tidParent);

        if (it != mapThreads_.end()) {
            parent = it->second;
        }

        // Set the private per-thread variables
#if 0
        thread *t = new thread(*this);
        ASSERTION(mapThreads_.find(util::get_thread_id()) == mapThreads_.end(),
                "Thread already registered");

        mapThreads_.insert(map_thread::value_type(util::get_thread_id(), t));

        gmacError_t err = resourceManager_.init_thread(*t, parent);
        ASSERTION(err == gmacSuccess);
#endif
    }
}

void process::fini_thread(bool userThread)
{
    // TODO: remove vdevices in virtual device table
    //if (Thread::hasCurrentVirtualDevice() != false) removevdevice(Thread::getCurrentVirtualDevice());

    if (userThread) {
        map_thread::iterator it = mapThreads_.find(util::get_thread_id());
        ASSERTION(it != mapThreads_.end(), "Thread not registered");
        mapThreads_.erase(it);

        delete &TLS::get_current_thread();
    }
}

}}}
