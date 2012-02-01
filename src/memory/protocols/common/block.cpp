#include "block.h"

#include "memory/object.h"

namespace __impl {
namespace memory { namespace protocols { namespace common {

const char *StatisticName[] = {
    "faults_read",
    "faults_write",
    "transfers_toacc",
    "transfers_tohost"
};

block::block(object &parent, size_t offset, size_t size) :
    Lock("block"),
    parent_(parent),
    size_(size),
    offset_(offset),
    faultsCacheWrite_(0),
    faultsCacheRead_(0)
{
}

const block::bounds
block::get_bounds() const
{
    // No need for lock -- addr_ is never modified
    object::bounds boundsParent = parent_.get_bounds();
    return bounds(boundsParent.start + offset_,
                  boundsParent.start + offset_ + size_);
}

size_t
block::size() const
{
    return size_;
}

host_ptr
block::get_shadow() const
{
    return parent_.get_bounds_shadow().start + offset_;
}

unsigned
block::get_faults_cache_write() const
{
    return faultsCacheWrite_;
}

unsigned
block::get_faults_cache_read() const
{
    return faultsCacheRead_;
}

void
block::reset_faults_cache_write()
{
    faultsCacheWrite_ = 0;
}

void
block::reset_faults_cache_read()
{
    faultsCacheRead_ = 0;
}

}}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
