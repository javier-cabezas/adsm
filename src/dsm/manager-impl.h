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
        (it->second->get_ptr().get_offset() >= addr.get_offset() + hal::ptr::offset_type(count))) {
        map_mapping::iterator it;
        return range_mapping(it, it);
    } else {
        begin = it;
    }

    // Add all the mappings in the range
    do {
        ++it;
    } while ((it != group.end()) &&
             (it->second->get_ptr().get_offset() < addr.get_offset() + hal::ptr::offset_type(count)));

    end = it;

    return range_mapping(begin, end);
}

template <bool All>
bool
manager::range_has_protection(const range_mapping &range, GmacProtection prot)
{
    ASSERTION(range.is_empty() == false, "Checking permissions for an empty range");

    if (All) {
        for (auto m : range) {
            if (m->get_protection() != prot) return false;
        }

        return true;
    } else {
        for (auto m : range) {
            if (m->get_protection() == prot) return true;
        }

        return false;
    }
}

template <bool Hex, bool PrintBlocks>
void
manager::range_print(const range_mapping &range)
{
    if (range.is_empty()) {
        printf("{ Empty }\n");
    } else {
        for (auto m : range) {
            hal::ptr::offset_type begin = m->get_ptr().get_offset();
            if (Hex) {
                printf(FMT_ID2 " [%p - %p]\n", m->get_print_id2(), (void *) begin, (void *) (begin + m->get_bounds().get_size()));
            } else {
                printf(FMT_ID2 " [%zd - %zd]\n", m->get_print_id2(), begin, begin + m->get_bounds().get_size());
            }
            if (PrintBlocks) {
                m->print<Hex>();
            }
        }
    }
}

template <bool Hex, bool PrintBlocks>
void
manager::print_all_mappings(hal::virt::aspace &as)
{
    map_mapping_group &mappings = get_aspace_mappings(as);
    range_mapping range(mappings.begin(), mappings.end());

    if (range.is_empty()) {
        printf("{ Empty }\n");
    } else {
        for (auto m : range) {
            hal::ptr::offset_type begin = m->get_ptr().get_offset();
            if (Hex) {
                printf(FMT_ID2 " [%p - %p]\n", m->get_print_id2(), (void *) begin, (void *) (begin + m->get_bounds().get_size()));
            } else {
                printf(FMT_ID2 " [%zd - %zd]\n", m->get_print_id2(), begin, begin + m->get_bounds().get_size());
            }
            if (PrintBlocks) {
                m->print<Hex>();
            }
        }
    }
}
}}

#endif
