#include "manager.h"

namespace __impl { namespace dsm {

void
manager::event_handler(hal::aspace &aspace, util::event::construct)
{
    map_mapping_group *ret = new map_mapping_group();
    aspace.set_attribute<map_mapping_group>(AttributeMappings_, ret);
}

void
manager::event_handler(hal::aspace &aspace, util::event::destruct)
{
    // Get the mappings from the address space, to avoid an extra map
    map_mapping_group *ret = aspace.get_attribute<map_mapping_group>(AttributeMappings_);
    ASSERTION(ret != NULL);
    delete ret;
}

manager::map_mapping_group &
manager::get_aspace_mappings(hal::aspace &ctx)
{
    map_mapping_group *ret;

    // Get the mappings from the address space, to avoid an extra map
    ret = ctx.get_attribute<map_mapping_group>(AttributeMappings_);
    ASSERTION(ret != NULL);

    return *ret;
}

bool
mapping_fits(manager::map_mapping &map, mapping_ptr m)
{
    manager::map_mapping::iterator it = map.upper_bound(m->get_ptr().get_offset());

    return it == map.end() || it->second->get_ptr().get_offset() >= (m->get_ptr().get_offset() + m->get_bounds().get_size());
}

gmacError_t
manager::insert_mapping(map_mapping_group &mappings, mapping_ptr m)
{
    map_mapping_group::iterator it = mappings.find(m->get_ptr().get_base());

    if (it == mappings.end()) {
        map_mapping map;
        map.insert(map_mapping::value_type(m->get_ptr().get_offset() + m->get_bounds().get_size(), m));
        mappings.insert(map_mapping_group::value_type(m->get_ptr().get_base(), map));
        TRACE(LOCAL, "Inserting mapping: "FMT_SIZE"-"FMT_SIZE, m->get_ptr().get_offset(),
                                                               m->get_ptr().get_offset() + m->get_bounds().get_size());
    } else {
        if (mapping_fits(it->second, m)) {
            it->second.insert(map_mapping::value_type(m->get_ptr().get_offset() + m->get_bounds().get_size(), m));
            TRACE(LOCAL, "Inserting mapping: "FMT_SIZE"-"FMT_SIZE, m->get_ptr().get_offset(),
                                                                   m->get_ptr().get_offset() + m->get_bounds().get_size());
        } else {
            //ASSERTION(mapping_fits(it->second, m) == true);
            TRACE(LOCAL, "NOT inserting mapping: "FMT_SIZE"-"FMT_SIZE, m->get_ptr().get_offset(),
                                                                       m->get_ptr().get_offset() + m->get_bounds().get_size());
            return gmacErrorInvalidValue;
        }
    }

    return gmacSuccess;
}

mapping_ptr
manager::merge_mappings(hal::ptr p, size_t count, range_mapping &range)
{
    ASSERTION(!range.is_empty());

    mapping_ptr ret = factory_mapping::create(*(*range.begin()));
    range_mapping::iterator it = range.begin();

    ++it;
    for (; it != range.end(); ++it) {
        gmacError_t err = ret->append(*it);
        if (err != gmacSuccess) break;
    }

    return ret;
}

manager::manager() :
    observer_construct(),
    observer_destruct(),
    AttributeMappings_(hal::aspace::register_attribute())
{
}

manager::~manager()
{
}

gmacError_t
manager::link(hal::ptr dst, hal::ptr src, size_t count, int flags)
{
    ASSERTION(long_t(dst.get_offset()) % mapping::MinAlignment == 0);
    ASSERTION(long_t(src.get_offset()) % mapping::MinAlignment == 0);

    hal::aspace *ctxDst = dst.get_aspace();
    hal::aspace *ctxSrc = src.get_aspace();

    map_mapping_group &mappingsDst = get_aspace_mappings(*ctxDst);
    map_mapping_group &mappingsSrc = get_aspace_mappings(*ctxSrc);

    range_mapping rangeDst = get_mappings_in_range<false>(mappingsDst, dst, count);
    range_mapping rangeSrc = get_mappings_in_range<false>(mappingsSrc, src, count);

    mapping_ptr mDst = merge_mappings(dst, count, rangeDst);
    mapping_ptr mSrc = merge_mappings(src, count, rangeSrc);

    mapping::submappings subDst, subSrc;

    gmacError_t ret = mapping::link(dst, mDst,
                                    src, mSrc, count, flags);

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
