#ifndef GMAC_DSM_MAPPING_IMPL_H_
#define GMAC_DSM_MAPPING_IMPL_H_

namespace __impl { namespace dsm {

inline
mapping::bounds
mapping::get_bounds() const
{
    bounds ret(addr_.get_offset(), addr_.get_offset() + size_);

    return ret;
}

inline
hal::ptr
mapping::get_ptr() const
{
    return addr_;
}

template <typename I>
mapping_ptr
mapping::merge_mappings(util::range<I> range, hal::ptr::offset_type off, size_t count)
{
    ASSERTION(range.is_empty() == false, "Merging an empty range");

    I it = range.begin;

    if ((*range.begin)->get_bounds().start > off) {
        // Grow first mapping upwards 
        // Create block for the memory preceeding any existing mapping
        size_t prefix = (*range.begin)->get_bounds().start - off;
        ASSERTION(count > prefix);

        coherence::block_ptr b = new coherence::block(prefix);
        (*range.begin)->prepend(b);
    } else if ((*range.begin)->get_bounds().start < off) {
        // Split the blocks within the first mapping if needed
        (*range.begin)->split(off, count);
    }

    // Merge the mappings into the first one
    ++it;

    for (; it != range.end; ++it) {
        ASSERTION((*range.begin)->addr_.get_base() ==
                           (*it)->addr_.get_base());
        (*range.begin)->append(*it);
    }

    return *range.begin;
}

template <typename I>
gmacError_t
mapping::link(hal::ptr ptr1, util::range<I> range1, submappings &sub1,
              hal::ptr ptr2, util::range<I> range2, submappings &sub2, size_t count, int flags)
{
    ASSERTION(long_t(ptr1.get_offset()) % MinAlignment == 0);
    ASSERTION(long_t(ptr2.get_offset()) % MinAlignment == 0);

    gmacError_t ret;

    I begin1 = range1.begin;
    I end1   = range1.end;
    I begin2 = range2.begin;
    I end2   = range2.end;

    // Case1
    // No previous links in both ranges: link new mappings
    if (begin1 == end1 && begin2 == end2) {
        mapping_ptr map1, map2;

        map1 = new mapping(ptr1);
        map2 = new mapping(ptr2);

        /// TODO: 
        coherence::block_ptr b = new coherence::block(count);

        ret = map1->prepend(b);
        ASSERTION(ret == gmacSuccess);
        ret = map2->prepend(b);
        ASSERTION(ret == gmacSuccess);

        sub1.push_back(map1);
        sub2.push_back(map2);
    } else if (begin1 == end1) {
        mapping_ptr map1, map2;
        map1 = new mapping(ptr1);
        map2 = merge_mappings(range2, ptr2.get_offset(), count);
        ret = map1->dup(ptr1.get_offset(), map2, ptr2.get_offset(), count);
    } else if (begin2 == end2) {
        mapping_ptr map1, map2;
        map2 = new mapping(ptr2);
        map1 = merge_mappings(range1, ptr1.get_offset(), count);
        ret = map2->dup(ptr2.get_offset(), map1, ptr1.get_offset(), count);
    } else {
        mapping_ptr map1, map2;
        map1 = merge_mappings(range1, ptr1.get_offset(), count);
        map2 = merge_mappings(range2, ptr2.get_offset(), count);
        ret = mapping::dup2(map1, ptr1.get_offset(),
                            map2, ptr2.get_offset(), count);
    }

    return ret;
}

}}

#endif /* GMAC_DSM_MAPPING_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
