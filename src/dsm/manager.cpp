#include "manager.h"

namespace __impl { namespace dsm {

void
manager::aspace_created(manager *m, hal::context_t &aspace)
{
    map_mapping_group *ret = new map_mapping_group();
    aspace.set_attribute<map_mapping_group>(m->AttributeMappings_, ret);
}

void
manager::aspace_destroyed(manager *m, hal::context_t &aspace)
{
    // Get the mappings from the address space, to avoid an extra map
    map_mapping_group *ret = aspace.get_attribute<map_mapping_group>(m->AttributeMappings_);
    ASSERTION(ret != NULL);
    delete ret;
}

manager::map_mapping_group &
manager::get_aspace_mappings(hal::context_t &ctx)
{
    map_mapping_group *ret;

    // Get the mappings from the address space, to avoid an extra map
    ret = ctx.get_attribute<map_mapping_group>(AttributeMappings_);
    ASSERTION(ret != NULL);

    return *ret;
}


manager::range_mapping
manager::get_mappings_in_range(map_mapping_group &mappings, hal::ptr addr, size_t count)
{
    map_mapping_group::iterator itGroup = mappings.find(addr.get_base());

    // If we don't find the base allocation for the address, return empty range
    if (itGroup == mappings.end()) {
        map_mapping::iterator it;
        return range_mapping(it, it);
    }

    map_mapping &group = *itGroup->second;
    map_mapping::iterator begin, end;
    map_mapping::iterator it = group.upper_bound(addr.get_offset());

    // If no mapping is affected, return empty range
    if (it == group.end() || it->second->get_ptr().get_offset() >= addr.get_offset() + count) {
        map_mapping::iterator it;
        return range_mapping(it, it);
    } else {
        begin = it;
    }

    // Add all the mappings in the range
    do {
        ++it;
    } while (it != group.end() && it->second->get_ptr().get_offset() < addr.get_offset() + count);

    end = it;

    return range_mapping(begin, end);
}

manager::manager() :
    AttributeMappings_(hal::context_t::register_attribute())
{
    hal::context_t::add_constructor(do_func(manager::aspace_created, this, std::placeholders::_1));
    hal::context_t::add_destructor(do_func(manager::aspace_destroyed, this, std::placeholders::_1));
}

manager::~manager()
{
}

gmacError_t
manager::link(hal::ptr dst, hal::ptr src, size_t count, int flags)
{
    ASSERTION(long_t(dst.get_offset()) % mapping::MinAlignment == 0);
    ASSERTION(long_t(src.get_offset()) % mapping::MinAlignment == 0);

    hal::context_t *ctxDst = dst.get_context();
    hal::context_t *ctxSrc = src.get_context();

    map_mapping_group &mappingsDst = get_aspace_mappings(*ctxDst);
    map_mapping_group &mappingsSrc = get_aspace_mappings(*ctxSrc);

    range_mapping rangeDst = get_mappings_in_range(mappingsDst, dst, count);
    range_mapping rangeSrc = get_mappings_in_range(mappingsSrc, src, count);

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
