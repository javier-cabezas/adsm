#include "config/order.h"
#include "core/address_space.h"
#include "util/FileSystem.h"

#include "map_object.h"
#include "object.h"
#include "protocol.h"

namespace __impl { namespace memory {

#ifdef DEBUG
Atomic map_object::StatsInit_ = 0;
Atomic map_object::StatDumps_ = 0;
std::string map_object::StatsDir_ = "";

void
map_object::statsInit()
{
    if (config::params::Stats) {
        PROCESS_T pid = __impl::util::GetProcessId();

        std::stringstream ss(std::stringstream::out);
#if defined(_MSC_VER)
        char tmpDir[256];
        GetTempPath(256, tmpDir);
        ss << tmpDir << "\\" << pid << "\\";
#else
        ss << ".gmac-" << pid << "/";
#endif
        bool created = __impl::util::MakeDir(ss.str());
        ASSERTION(created == true);
        StatsDir_ = ss.str();
    }
}
#endif

object_ptr
map_object::map_find(const hostptr_t addr, size_t size) const
{
    map_object::const_iterator i;
    object_ptr ret;
    lock_read();
    const uint8_t *limit = (const uint8_t *)addr + size;
    i = upper_bound(addr);
    if(i != end() && i->second->get_bounds().start <= limit) {
    	ret = object_ptr(i->second);
    }
    unlock();
    return ret;
}

map_object::map_object(const char *name) :
    Lock(name),
    protocol_(*ProtocolInit(0)),
    modifiedObjects_(false),
    releasedObjects_(false)
#ifdef USE_VM
    , bitmap_(*this)
#endif
{

#ifdef DEBUG
    if(AtomicTestAndSet(StatsInit_, 0, 1) == 0) statsInit();
#endif
}

map_object::~map_object()
{
}

void
map_object::cleanUp()
{
    lock_write();
    clear();
    unlock();
}

size_t
map_object::size() const
{
    lock_read();
    size_t ret = Parent::size();
    unlock();
    return ret;
}

#if 0
core::Process &
map_object::getProcess()
{
    return parent_;
}

const core::Process &
map_object::getProcess() const
{
    return parent_;
}
#endif

bool
map_object::add_object(object &obj)
{
    lock_write();
    TRACE(LOCAL, "Insert object: %p", obj.get_bounds().start);
    object_ptr ptr(&obj);
    std::pair<iterator, bool> ret = Parent::insert(value_type(obj.get_bounds().end, ptr));
    modifiedObjects_unlocked();
    unlock();
    return ret.second;
}

bool
map_object::remove_object(object &obj)
{
    lock_write();
    iterator i = find(obj.get_bounds().end);
    bool ret = (i != end());
    if(ret == true) {
#if defined(DEBUG)
        if (config::params::Stats) {
            unsigned dump = AtomicInc(StatDumps_);
            std::stringstream ss(std::stringstream::out);
            ss << dump << "-" << "remove";

            dumpObject(StatsDir_, ss.str(), memory::protocols::common::PAGE_FAULTS_READ, obj.get_bounds().start);
            dumpObject(StatsDir_, ss.str(), memory::protocols::common::PAGE_FAULTS_WRITE, obj.get_bounds().start);
            dumpObject(StatsDir_, ss.str(), memory::protocols::common::PAGE_TRANSFERS_TO_HOST, obj.get_bounds().start);
            dumpObject(StatsDir_, ss.str(), memory::protocols::common::PAGE_TRANSFERS_TO_ACCELERATOR, obj.get_bounds().start);
        }
#endif

        TRACE(LOCAL, "Remove object: %p", obj.get_bounds().start);
        Parent::erase(i);
    } else {
        TRACE(LOCAL, "CANNOT Remove object: %p from map with "FMT_SIZE" elems", obj.get_bounds().start, Parent::size());
    }
    unlock();
    return ret;
}

object_ptr
map_object::get_object(const hostptr_t addr, size_t size) const
{
    // Lock already acquired in map_find
    return map_find(addr, size);
}

size_t
map_object::get_memory_size() const
{
    size_t total = 0;
    const_iterator i;
    lock_read();
    for(i = begin(); i != end(); i++) {
        total += i->second->size();
    }
    unlock();
    return total;
}

gmacError_t
map_object::release_objects()
{
    lock_write();
#ifdef DEBUG
    if (config::params::Stats) {
        unsigned dump = AtomicInc(StatDumps_);
        std::stringstream ss(std::stringstream::out);
        ss << dump << "-" << "release";

        dumpObjects(StatsDir_, ss.str(), memory::protocols::common::PAGE_FAULTS_READ);
        dumpObjects(StatsDir_, ss.str(), memory::protocols::common::PAGE_FAULTS_WRITE);
        dumpObjects(StatsDir_, ss.str(), memory::protocols::common::PAGE_TRANSFERS_TO_HOST);
        dumpObjects(StatsDir_, ss.str(), memory::protocols::common::PAGE_TRANSFERS_TO_ACCELERATOR);
    }
#endif
    releasedObjects_ = true;
    unlock();
    return gmacSuccess;
}

gmacError_t
map_object::acquire_objects()
{
    lock_write();
    modifiedObjects_ = false;
    releasedObjects_ = false;
    unlock();
    return gmacSuccess;
}


}}
