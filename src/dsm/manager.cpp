#include <sstream>

#include "manager.h"

namespace __impl { namespace dsm {

static
bool TRUE = true;
static
bool FALSE = true;

void
manager::event_handler(hal::virt::aspace &as, const util::event::construct &)
{
    TRACE(LOCAL, FMT_ID2" Handle " FMT_ID2" creation", get_print_id2(), as.get_print_id2());

    map_mapping_group *ret = new map_mapping_group();
    as.set_attribute<map_mapping_group>(AttributeMappings_, ret);
    as.set_attribute<bool>(AttributeProtection_, &FALSE);
}

void
manager::event_handler(hal::virt::aspace &as, const util::event::destruct &)
{
    TRACE(LOCAL, FMT_ID2" Handle " FMT_ID2" destruction", get_print_id2(), as.get_print_id2());

    // Get the mappings from the address space, to avoid an extra map
    map_mapping_group &group = get_aspace_mappings(as);
    error err = delete_mappings(group);
    ASSERTION(err == error::DSM_SUCCESS);

    delete &group;
}

bool
manager::handle_fault(hal::ptr p, bool isWrite)
{
    hal::virt::aspace &as = p.get_view().get_vaspace();
    manager *mgr = manager::get_instance();

    // Retrieve the mappings for the address space
    map_mapping_group &mappings = mgr->get_aspace_mappings(as);
    // Retrieve the mappings for the view
    map_mapping_group::iterator it = mappings.find(&p.get_view());

    if (it == mappings.end()) {
        return false;
    }

    // Retrieve the mapping
    map_mapping &map = it->second;
    map_mapping::iterator it2 = map.upper_bound(p.get_offset());
    if (it2 == map.end()) {
        return false;
    }

    error err = it2->second->handle_fault(p - it2->second->get_ptr(), isWrite);

    return err == error::DSM_SUCCESS;
}

error
manager::use_memory_protection(hal::virt::aspace &as)
{
    bool overloaded = as.get_attribute<bool>(AttributeProtection_);
    if (!overloaded) {
        hal::error errHal;
        hal::virt::handler_sigsegv handler(handle_fault);
        errHal = as.handler_sigsegv_push(handler);
        if (errHal != hal::error::HAL_SUCCESS) return error::DSM_ERROR_HAL;
        as.set_attribute<bool>(AttributeProtection_, &TRUE);
    }
    return error::DSM_SUCCESS;
}

manager::map_mapping_group &
manager::get_aspace_mappings(hal::virt::aspace &aspace)
{
    map_mapping_group *ret;

    // Get the mappings from the address space, to avoid an extra map
    ret = aspace.get_attribute<map_mapping_group>(AttributeMappings_);
    ASSERTION(ret != nullptr);

    return *ret;
}

bool
mapping_fits(manager::map_mapping &map, mapping_ptr m)
{
    manager::map_mapping::iterator it = map.upper_bound(m->get_ptr().get_offset());

    return it == map.end() || it->second->get_ptr().get_offset() >= hal::ptr::offset_type(m->get_ptr().get_offset() + m->get_bounds().get_size());
}

error
manager::insert_mapping(map_mapping_group &mappings, mapping_ptr m)
{
    map_mapping_group::iterator it = mappings.find(&m->get_ptr().get_view());

    if (it == mappings.end()) {
        map_mapping map;
        map.insert(map_mapping::value_type(m->get_ptr().get_offset() + m->get_bounds().get_size(), m));
        mappings.insert(map_mapping_group::value_type(&m->get_ptr().get_view(), map));
        TRACE(LOCAL, FMT_ID2" Inserting " FMT_ID2" in " FMT_ID2,
                     get_print_id2(),
                     m->get_print_id2(),
                     m->get_ptr().get_view().get_vaspace().get_print_id2());
    } else {
        map_mapping &map = it->second;
        if (mapping_fits(map, m)) {
            map.insert(map_mapping::value_type(m->get_ptr().get_offset() + m->get_bounds().get_size(), m));
            TRACE(LOCAL, FMT_ID2" Inserting " FMT_ID2" in " FMT_ID2,
                         get_print_id2(),
                         m->get_print_id2(),
                         m->get_ptr().get_view().get_vaspace().get_print_id2());
        } else {
            TRACE(LOCAL, FMT_ID2" NOT Inserting " FMT_ID2" in " FMT_ID2,
                         get_print_id2(),
                         m->get_print_id2(),
                         m->get_ptr().get_view().get_vaspace().get_print_id2());
            return error::DSM_ERROR_INVALID_VALUE;
        }
    }

    return error::DSM_SUCCESS;
}

template <typename R>
static std::string
get_range_string(const R &range)
{
    bool first = true;
    std::ostringstream s;
    for (auto r : range) {
        if (first) {
            first = 0;
            s << r->get_print_id2();
        } else {
            s << ", " << r->get_print_id2();
        }
    }

    return s.str();
}

mapping_ptr
manager::merge_mappings(range_mapping &range)
{
    TRACE(LOCAL, FMT_ID2" Merging range (%s)", get_print_id2(), get_range_string(range).c_str());

    ASSERTION(!range.is_empty(), "Merging an empty range of mappings");

    range_mapping::iterator it = range.begin();
    mapping_ptr ret = factory_mapping::create(std::move(**it));

    ++it;
    for (; it != range.end(); ++it) {
        error err = ret->append(std::move(**it));
        if (err != error::DSM_SUCCESS) break;
    }

    return ret;
}

error
manager::replace_mappings(map_mapping_group &mappings, range_mapping &range, mapping_ptr mNew)
{
    TRACE(LOCAL, FMT_ID2" Replace range (%s) with " FMT_ID2,
                 get_print_id2(),
                 get_range_string(range).c_str(),
                 mNew->get_print_id2());

    ASSERTION(!range.is_empty(), "Replacing an empty range of mappings");

    ASSERTION(&(*range.begin())->get_ptr().get_view() == &mNew->get_ptr().get_view());

    map_mapping_group::iterator it = mappings.find(&mNew->get_ptr().get_view());
    ASSERTION(it != mappings.end());

    it->second.erase(range.begin().base(), range.end().base());

    ASSERTION(get_mappings_in_range<false>(mappings, mNew->get_ptr(), mNew->get_bounds().get_size()).is_empty() == true);
    it->second.insert(map_mapping::value_type(mNew->get_ptr().get_offset(), mNew));

    return error::DSM_SUCCESS;
}

error
manager::delete_mappings(map_mapping_group &mappings)
{
    TRACE(LOCAL, FMT_ID2" Deleting ALL mappings", get_print_id2());

    for (auto &pairGroup : mappings) {
        for (auto pairMapping : pairGroup.second) {
            delete pairMapping.second;
        }
        pairGroup.second.clear();
    }
    mappings.clear();

    return error::DSM_SUCCESS;
}

manager::manager() :
    observer_construct(),
    observer_destruct(),
    AttributeMappings_(hal::virt::aspace::register_attribute()),
    AttributeProtection_(hal::virt::aspace::register_attribute())
{
    TRACE(LOCAL, FMT_ID2" Creating", get_print_id2());
}

manager::~manager()
{
    TRACE(LOCAL, FMT_ID2" Deleting", get_print_id2());
}

error
manager::link(hal::ptr dst, hal::ptr src, size_t count, GmacProtection protDst, GmacProtection protSrc, int flags)
{
    TRACE(LOCAL, FMT_ID2" Link " FMT_SIZE" bytes", get_print_id2(), count);

    // Pointers must be valid
    CHECK(bool(dst) && bool(src), error::DSM_ERROR_INVALID_PTR);
    // Pointers must belong to different address spaces (WHY!)
    CHECK(&dst.get_view().get_vaspace() != &src.get_view().get_vaspace(), error::DSM_ERROR_INVALID_PTR);

    // Alignment checks
    CHECK(long_t(dst.get_offset()) % mapping::MinAlignment == 0, error::DSM_ERROR_INVALID_ALIGNMENT);
    CHECK(long_t(src.get_offset()) % mapping::MinAlignment == 0, error::DSM_ERROR_INVALID_ALIGNMENT);

    // Link size must be greater than 0
    CHECK(count > 0, error::DSM_ERROR_INVALID_VALUE);

    // TODO: do something with the flags

    hal::virt::aspace &asDst = dst.get_view().get_vaspace();
    hal::virt::aspace &asSrc = src.get_view().get_vaspace();

    map_mapping_group &mappingsDst = get_aspace_mappings(asDst);
    map_mapping_group &mappingsSrc = get_aspace_mappings(asSrc);

    range_mapping rangeDst = get_mappings_in_range<false>(mappingsDst, dst, count);
    range_mapping rangeSrc = get_mappings_in_range<false>(mappingsSrc, src, count);

    mapping_ptr mDst;
    mapping_ptr mSrc;

    if (rangeDst.is_empty()) {
        // If the range is NOT used yet, create a new mapping
        mDst = factory_mapping::create(dst, protDst, flags);
    } else {
        if (range_has_protection<true>(rangeDst, protDst)) {
            // If the range is already used, merge previous mappings
            mDst = merge_mappings(rangeDst);
        } else {
            NOT_IMPLEMENTED();
        }
    }

    if (rangeSrc.is_empty()) {
        // If the range is NOT used yet, create a new mapping
        mSrc = factory_mapping::create(src, protSrc, flags);
    } else {
        if (range_has_protection<true>(rangeSrc, protSrc)) {
            // If the range is already used, merge previous mappings
            mSrc = merge_mappings(rangeSrc);
        } else {
            NOT_IMPLEMENTED();
        }
    }

    if (dst < mDst->get_ptr() ||
        mDst->get_bounds().get_size() < count) {
        // If the mapping is not big enoug, make it larger
        size_t pre, post;
        pre  = mDst->get_ptr().get_offset() - dst.get_offset();
        post = count - (mDst->get_bounds().get_size() + pre);
        mDst->resize(pre, post); 
    }
    if (src < mSrc->get_ptr() ||
        mSrc->get_bounds().get_size() < count) {
        // If the mapping is not big enoug, make it larger
        size_t pre, post;
        pre  = mSrc->get_ptr().get_offset() - src.get_offset();
        post = count - (mSrc->get_bounds().get_size() + pre);
        mSrc->resize(pre, post); 
    }

    // Perform the linking between the mappings
    error ret = mapping::link(dst.get_offset() - mDst->get_ptr().get_offset(), mDst,
                              src.get_offset() - mSrc->get_ptr().get_offset(), mSrc, count, 0);

    if (ret == error::DSM_SUCCESS) {
        if (rangeDst.is_empty()) {
            // If the mapping was new, insert it
            insert_mapping(mappingsDst, mDst);
        } else {
            // If mappings were reused, replace ...
            TRACE(LOCAL, FMT_ID2" Destroy range (%s)", get_print_id2(), get_range_string(rangeDst).c_str());
            range_mapping::list listDst = rangeDst.to_list();
            replace_mappings(mappingsDst, rangeDst, mDst);

            // ... and destroy them in the map
            for (mapping_ptr p : listDst) {
                delete p;
            }
        }
        if (rangeSrc.is_empty()) {
            // If the mapping was new, insert it
            insert_mapping(mappingsSrc, mSrc);
        } else {
            // If mappings were reused, replace ...
            TRACE(LOCAL, FMT_ID2" Destroy range (%s)", get_print_id2(), get_range_string(rangeSrc).c_str());
            range_mapping::list listSrc = rangeSrc.to_list();
            replace_mappings(mappingsSrc, rangeSrc, mSrc);

            // ... and destroy them in the map
            for (mapping_ptr p : listSrc) {
                delete p;
            }
        }
    }

    return ret;
}

error
manager::unlink(hal::ptr mapping, size_t count)
{
    TRACE(LOCAL, FMT_ID2" Unlink requested", get_print_id2());

    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::acquire(hal::ptr p, size_t count, GmacProtection prot)
{
    error ret = error::DSM_SUCCESS;

    hal::virt::aspace &as = p.get_view().get_vaspace();

    map_mapping_group &mappings = get_aspace_mappings(as);
 
    range_mapping range = get_mappings_in_range<false>(mappings, p, count);

    for (auto m : range) {
        size_t off = 0;
        size_t local;
        if (p > m->get_ptr()) {
            off = p - m->get_ptr();
        }
        if (count > (m->get_bounds().get_size() - off)) {
            local = m->get_bounds().get_size() - off;
        } else {
            local = count;
        }
        ret = m->acquire(off, local, prot);
        if (ret != error::DSM_SUCCESS) break;
        count -= local;
    }
    ASSERTION(count == 0);

    return ret;
}

error
manager::release(hal::ptr p, size_t count)
{
    error ret = error::DSM_SUCCESS;

    hal::virt::aspace &as = p.get_view().get_vaspace();

    map_mapping_group &mappings = get_aspace_mappings(as);
 
    range_mapping range = get_mappings_in_range<false>(mappings, p, count);

    for (auto m : range) {
        size_t off = 0;
        size_t local;
        if (p > m->get_ptr()) {
            off = p - m->get_ptr();
        }
        if (count > (m->get_bounds().get_size() - off)) {
            local = m->get_bounds().get_size() - off;
        } else {
            local = count;
        }
        ret = m->release(off, local);
        if (ret != error::DSM_SUCCESS) break;
        count -= local;
    }
    ASSERTION(count == 0);

    return ret;
}

error
manager::sync(hal::ptr mapping, size_t count)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::memcpy(hal::ptr dst, hal::ptr src, size_t count)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::memset(hal::ptr ptr, int c, size_t count)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::from_io_device(hal::ptr addr, hal::device_input &input, size_t count)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::to_io_device(hal::device_output &output, hal::const_ptr addr, size_t count)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

error
manager::flush_dirty(address_space_ptr aspace)
{
    FATAL("Not implemented");
    return error::DSM_SUCCESS;
}

}}
