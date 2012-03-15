#ifndef GMAC_DSM_MANAGER_IMPL_H_
#define GMAC_DSM_MANAGER_IMPL_H_

namespace __impl { namespace dsm {

template <bool GetAdjacent>
manager::range_mapping
manager::get_mappings_in_range(map_mapping_group &mappings, hal::ptr addr, size_t count)
{
    ASSERTION(count > 0);
    ASSERTION(addr);

    if (GetAdjacent) {
        ++count;
        if (addr.get_offset() > 0) {
            ++count;
            addr -= 1;
        }
    }

    map_mapping_group::iterator itGroup = mappings.find(&addr.get_view());

    // If we don't find the base allocation for the address, return empty range
    if (itGroup == mappings.end()) {
        map_mapping::iterator it;
        return range_mapping(it, it);
    }

    map_mapping &group = itGroup->second;
    map_mapping::iterator begin, end;
    map_mapping::iterator it = group.upper_bound(addr.get_offset());

    // If no mapping is affected, return empty range
    if (it == group.end() ||
        (it->second->get_ptr().get_offset() >= addr.get_offset() + count)) {
        map_mapping::iterator it;
        return range_mapping(it, it);
    } else {
        begin = it;
    }

    // Add all the mappings in the range
    do {
        ++it;
    } while (it != group.end() && (it->second->get_ptr().get_offset() < addr.get_offset() + count));

    end = it;

    return range_mapping(begin, end);
}

}}

#endif
