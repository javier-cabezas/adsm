#include "manager.h"

namespace __impl { namespace dsm {

void
manager::event_handler(hal::aspace &aspace, util::event::construct)
{
    TRACE(LOCAL, "Handling aspace creation");

    map_mapping_group *ret = new map_mapping_group();
    aspace.set_attribute<map_mapping_group>(AttributeMappings_, ret);
}

void
manager::event_handler(hal::aspace &aspace, util::event::destruct)
{
    TRACE(LOCAL, "Handling aspace destruction");

    // Get the mappings from the address space, to avoid an extra map
    map_mapping_group *ret = aspace.get_attribute<map_mapping_group>(AttributeMappings_);
    ASSERTION(ret != NULL);
    for (map_mapping_group::iterator it  = ret->begin();
                                     it != ret->end();
                                   ++it) {
        
        for (map_mapping::iterator it2  = it->second.begin();
                                   it2 != it->second.end();
                                 ++it2) {
            delete it2->second;
        }
    }

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

error
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
            TRACE(LOCAL, "NOT inserting mapping: "FMT_SIZE"-"FMT_SIZE, m->get_ptr().get_offset(),
                                                                       m->get_ptr().get_offset() + m->get_bounds().get_size());
            return DSM_ERROR_INVALID_VALUE;
        }
    }

    return DSM_SUCCESS;
}

mapping_ptr
manager::merge_mappings(range_mapping &range)
{
    ASSERTION(!range.is_empty());

    mapping_ptr ret = factory_mapping::create(*(*range.begin()));
    range_mapping::iterator it = range.begin();

    ++it;
    for (; it != range.end(); ++it) {
        error err = ret->append(*it);
        if (err != DSM_SUCCESS) break;
    }

    return ret;
}

error
manager::replace_mappings(map_mapping_group &mappings, range_mapping &range, mapping_ptr mNew)
{
    ASSERTION(!range.is_empty());

    ASSERTION((*range.begin())->get_ptr().get_base() == mNew->get_ptr().get_base());
    ASSERTION((*range.begin())->get_ptr().get_aspace() == mNew->get_ptr().get_aspace());

    map_mapping_group::iterator it = mappings.find(mNew->get_ptr().get_base());
    ASSERTION(it != mappings.end());

    it->second.erase(range.begin().base(), range.end().base());
    it->second.insert(map_mapping::value_type(mNew->get_ptr().get_offset(), mNew));

    return DSM_SUCCESS;
}

manager::manager() :
    observer_construct(),
    observer_destruct(),
    AttributeMappings_(hal::aspace::register_attribute())
{
    TRACE(LOCAL, "Creating DSM manager");
}

manager::~manager()
{
    TRACE(LOCAL, "Deleting DSM manager");
}

error
manager::link(hal::ptr dst, hal::ptr src, size_t count, int flags)
{
    CHECK(long_t(dst.get_offset()) % mapping::MinAlignment == 0, DSM_ERROR_INVALID_ALIGNMENT);
    CHECK(long_t(src.get_offset()) % mapping::MinAlignment == 0, DSM_ERROR_INVALID_ALIGNMENT);

    hal::aspace *ctxDst = dst.get_aspace();
    hal::aspace *ctxSrc = src.get_aspace();

    map_mapping_group &mappingsDst = get_aspace_mappings(*ctxDst);
    map_mapping_group &mappingsSrc = get_aspace_mappings(*ctxSrc);

    range_mapping rangeDst = get_mappings_in_range<false>(mappingsDst, dst, count);
    range_mapping rangeSrc = get_mappings_in_range<false>(mappingsSrc, src, count);

    mapping_ptr mDst;
    mapping_ptr mSrc;

    if (rangeDst.is_empty()) {
        mDst = factory_mapping::create(dst);
    } else {
        mDst = merge_mappings(rangeDst);
    }

    if (rangeSrc.is_empty()) {
        mSrc = factory_mapping::create(src);
    } else {
        mSrc = merge_mappings(rangeSrc);
    }

    if (dst < mDst->get_ptr() ||
        mDst->get_bounds().get_size() < count) {
        size_t pre, post;
        pre  = mDst->get_ptr().get_offset() - dst.get_offset();
        post = count - (mDst->get_bounds().get_size() + pre);
        mDst->resize(pre, post); 
    }
    if (src < mSrc->get_ptr() ||
        mSrc->get_bounds().get_size() < count) {
        size_t pre, post;
        pre  = mSrc->get_ptr().get_offset() - src.get_offset();
        post = count - (mSrc->get_bounds().get_size() + pre);
        mSrc->resize(pre, post); 
    }

    error ret = mapping::link(dst, mDst,
                              src, mSrc, count, flags);

    if (ret == DSM_SUCCESS) {
        if (rangeDst.is_empty()) {
            insert_mapping(mappingsDst, mDst);
        } else {
            replace_mappings(mappingsDst, rangeDst, mDst);
        }
        if (rangeSrc.is_empty()) {
            insert_mapping(mappingsSrc, mSrc);
        } else {
            replace_mappings(mappingsSrc, rangeSrc, mSrc);
        }
    }

    return ret;
}

error
manager::unlink(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::acquire(hal::ptr mapping, size_t count, int flags)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::release(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::sync(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::memcpy(hal::ptr dst, hal::ptr src, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::memset(hal::ptr ptr, int c, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::from_io_device(hal::ptr addr, hal::device_input &input, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::to_io_device(hal::device_output &output, hal::const_ptr addr, size_t count)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

error
manager::flush_dirty(address_space_ptr aspace)
{
    FATAL("Not implemented");
    return DSM_SUCCESS;
}

}}
