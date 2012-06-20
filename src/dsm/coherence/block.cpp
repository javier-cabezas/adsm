#include "dsm/manager.h"

#include "block.h"

namespace __impl { namespace dsm { namespace coherence {

error
block::update(mapping_ptr m, size_t off)
{
    // Find the mapping holding the last copy of the block
    auto it = util::algo::find_if(mappings_,
                                  [](mappings::value_type &val)
                                  {
                                      if (val.second.state_ != state::STATE_INVALID) return true;
                                      return false;
                                  });
    
    // Someone must have the last copy
    CHECK(it != mappings_.end(), error::DSM_ERROR_PROTOCOL);

    // Only update if mappings are different
    if (m != it->first) {
        hal::error errHal;
        hal::copy(        m->get_ptr() + off, 
                  it->first->get_ptr() + it->second.off_,
                  size_, errHal);

        if (errHal != hal::error::HAL_SUCCESS) {
            return error::DSM_ERROR_HAL;
        }
    }

    return error::DSM_SUCCESS;
}

error
block::acquire(mapping_ptr m, GmacProtection prot)
{
    TRACE(LOCAL, FMT_ID2" Acquire block for " FMT_ID2, get_print_id2(), m->get_print_id2());

    ASSERTION(prot <= m->get_protection());

    CHECK(m != owner_, error::DSM_ERROR_OWNERSHIP);

    error ret = error::DSM_SUCCESS;

    auto it = mappings_.find(m);

    if (it->second.state_ == state::STATE_INVALID) {
        // Invalid, but no other copies
        CHECK(mappings_.size() > 1, error::DSM_ERROR_PROTOCOL);

        if ((prot_is_writable(prot) || prot_is_readable(prot)) &&
            has_mapping_flag(m->get_flags(), mapping_flags::MAP_USE_PROTECT)) {
            // Use protection for lazy-update
            hal::error errHal;
            errHal = m->get_ptr().get_view().get_vaspace().protect(m->get_ptr() + it->second.off_, size_, GMAC_PROT_NONE);

            dsm::manager *mgr = dsm::manager::get_instance();
            ret = mgr->use_memory_protection(m->get_aspace());

            if (errHal == hal::error::HAL_SUCCESS) {
                return error::DSM_ERROR_HAL;
            }
        } else {
            // We need to update the contents
            ret = update(m, it->second.off_);

            it->second.state_ = state::STATE_SHARED;
        }
    }

    owner_ = m;
    prot_  = prot;

#if 0
    if (flags & GMAC_PROT_WRITE) {
        lock::lock_write();
    } else if (flags & GMAC_PROT_READ) {
        lock::lock_read();
    } else {
        FATAL("Wrong GmacProtection flag for acquire");
        return error::DSM_ERROR_INVALID_PROT;
    }
#endif

    return ret;
}

error
block::release(mapping_ptr m)
{
    TRACE(LOCAL, FMT_ID2" Release block from " FMT_ID2, get_print_id2(), m->get_print_id2());

    CHECK(m == owner_, error::DSM_ERROR_OWNERSHIP);

    if (prot_is_writable(prot_)) {
        // Invalidate everybody else
        for (auto &md : mappings_) {
            if (md.first != m) {
                TRACE(LOCAL, FMT_ID2" Set " FMT_ID2 " invalid", get_print_id2(), md.first->get_print_id2());
                md.second.state_ = state::STATE_INVALID;
            } else {
                // Set state to dirty
                // TODO: use memory protection to detect changes
                TRACE(LOCAL, FMT_ID2" Set " FMT_ID2 " dirty", get_print_id2(), md.first->get_print_id2());
                md.second.state_ = state::STATE_DIRTY;
            }
        }
    }

    // Unset owner
    owner_ = nullptr;

#if 0
    lock::unlock();
#endif

    return error::DSM_SUCCESS;
}

}}}
