#include "manager.h"

namespace __impl { namespace dsm {

void
manager::aspace_created(manager *m, hal::context_t &aspace)
{
    map_mappings *ret = new map_mappings;
    aspace.set_attribute<map_mappings>(m->AttributeMappings_, ret);
}

void
manager::aspace_destroyed(manager *m, hal::context_t &aspace)
{
    map_mappings *ret = aspace.get_attribute<map_mappings>(m->AttributeMappings_);
    ASSERTION(ret != NULL);
}

manager::map_mappings &
manager::get_aspace_mappings(hal::context_t &ctx)
{
    map_mappings *ret;

    ret = ctx.get_attribute<map_mappings>(AttributeMappings_);
    ASSERTION(ret != NULL);

    return *ret;
}


manager::range_mapping
manager::get_mappings_in_range(map_mappings &mappings, hal::ptr::address addr, size_t count)
{
    map_mappings::iterator begin, end;
    map_mappings::iterator it = mappings.upper_bound(addr);

    if (it == mappings.end() || it->second->get_bounds().start >= addr + count) {
        return range_mapping(mappings.end(), mappings.end());
    } else {
        begin = it;
    }

    do {
        ++it;
    } while (it != mappings.end() && it->second->get_bounds().start < addr + count);

    end = it;

    return range_mapping(begin, end);
}

manager::manager() :
    AttributeMappings_(hal::context_t::register_attribute())
{
    hal::context_t::add_constructor(do_func(manager::aspace_created, this, std::placeholders::_1));
    hal::context_t::add_destructor(do_func(manager::aspace_destroyed, this, std::placeholders::_1));
}

gmacError_t
manager::link(hal::ptr dst, hal::ptr src, size_t count, int flags)
{
    ASSERTION(long_t(dst.get_addr()) % mapping::MinAlignment == 0);
    ASSERTION(long_t(src.get_addr()) % mapping::MinAlignment == 0);

    hal::context_t *ctxDst = dst.get_context();
    hal::context_t *ctxSrc = src.get_context();

    map_mappings &mappingsDst = get_aspace_mappings(*ctxDst);
    map_mappings &mappingsSrc = get_aspace_mappings(*ctxSrc);

    range_mapping rangeDst = get_mappings_in_range(mappingsDst, dst.get_addr(), count);
    range_mapping rangeSrc = get_mappings_in_range(mappingsSrc, src.get_addr(), count);

    mapping::submappings subDst, subSrc;

    gmacError_t ret = mapping::link(dst, rangeDst, subDst, src, rangeSrc, subSrc, count, flags);

    return ret;
}

gmacError_t
manager::unlink(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::acquire(hal::ptr mapping, size_t count, int flags)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::release(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::sync(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::memcpy(hal::ptr dst, hal::ptr src, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::memset(hal::ptr ptr, int c, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::from_io_device(hal::ptr addr, hal::device_input &input, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::to_io_device(hal::device_output &output, hal::const_ptr addr, size_t count)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

gmacError_t
manager::flush_dirty(address_space_ptr aspace)
{
    FATAL("Not implemented");
    return gmacSuccess;
}

}}
