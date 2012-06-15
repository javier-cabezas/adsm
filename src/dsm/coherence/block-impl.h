#ifndef GMAC_DSM_COHERENCE_BLOCK_IMPL_H_
#define GMAC_DSM_COHERENCE_BLOCK_IMPL_H_

namespace __impl { namespace dsm { namespace coherence {

inline
block::block(size_t size) :
    lock("block"),
    size_(size),
    owner_(nullptr)
{
    TRACE(LOCAL, FMT_ID2" Creating " FMT_SIZE" bytes", get_print_id2(), size);
}

inline
block::~block()
{
    TRACE(LOCAL, FMT_ID2" Deleting", get_print_id2());
}

inline
block_ptr
block::split(size_t off)
{
    TRACE(LOCAL, FMT_ID2" Splitting " FMT_SIZE, get_print_id2(), off);

    // Create new block
    block_ptr nBlock = block_ptr(new block(size_ - off));
    // Set new size
    size_ = off;

    for (auto m : mappings_) {
        auto ret = nBlock->mappings_.insert(mappings::value_type(m.first, m.second));
        ASSERTION(ret.second == true);
        // Update the offset within mapping for the new block
        ret.first->second.off_ += off;
        m.first->block_splitted<false>(shared_from_this(), nBlock);
    }

    return block_ptr(nBlock);
}

inline
void
block::shift(mapping_ptr m, size_t off)
{
    TRACE(LOCAL, FMT_ID2" Shift " FMT_SIZE" bytes", get_print_id2(), off);

    mappings::iterator it = mappings_.find(m);
    ASSERTION(it != mappings_.end());

    it->second.off_ += off;
}

inline
error
block::acquire(mapping_ptr m, GmacProtection prot)
{
    TRACE(LOCAL, FMT_ID2" Acquire block for " FMT_ID2, get_print_id2(), m->get_print_id2());

    ASSERTION(prot <= m->get_protection());

    CHECK(m != owner_, error::DSM_ERROR_OWNERSHIP);

    auto it = mappings_.find(m);

    if (it->second.state_ == state::STATE_INVALID) {
        // Invalid, but no other copies
        CHECK(mappings_.size() > 1, error::DSM_ERROR_PROTOCOL);

        // We need to update the contents
        // Find the last copy
        auto it2 = util::algo::find_if(mappings_,
                                       [=](mappings::value_type &val)
                                       {
                                           if (val.second.state_ != state::STATE_INVALID) return true;
                                           return false;
                                       });
        
        // Someone must have the last copy
        CHECK(it2 != mappings_.end(), error::DSM_ERROR_PROTOCOL);

        hal::error err;
        hal::copy(         m->get_ptr() + it->second.off_, 
                  it2->first->get_ptr() + it2->second.off_,
                  size_, err);

        it->second.state_ = state::STATE_SHARED;
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

    return error::DSM_SUCCESS;
}

inline
error
block::release(mapping_ptr m)
{
    TRACE(LOCAL, FMT_ID2" Release block from " FMT_ID2, get_print_id2(), m->get_print_id2());

    CHECK(m == owner_, error::DSM_ERROR_OWNERSHIP);

    if (prot_is_writable(prot_)) {
        // Invalidate everybody else
        for (auto &md : mappings_) {
            if (md.first != m) {
                md.second.state_ = state::STATE_INVALID;
		TRACE(LOCAL, FMT_ID2" Set " FMT_ID2 " invalid", get_print_id2(), md.first->get_print_id2());
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

inline
error
block::register_mapping(mapping_ptr m, size_t off)
{
    TRACE(LOCAL, FMT_ID2" Register " FMT_ID2, get_print_id2(), m->get_print_id2());
    ASSERTION(mappings_.find(m) == mappings_.end(), "Mapping already registered");

    state s = state::STATE_INVALID;

    if (mappings_.size() == 0) {
         s = state::STATE_SHARED;
    }

    mapping_descriptor descr = {
                                   off,
                                   s
                               };
    mappings_.insert(mappings::value_type(m, descr));
    return error::DSM_SUCCESS;
}

inline
error
block::unregister_mapping(mapping &m)
{
    TRACE(LOCAL, FMT_ID2" Unregister " FMT_ID2, get_print_id2(), m.get_print_id2());
    CHECK(&m != owner_, error::DSM_ERROR_OWNERSHIP);

    mappings::iterator it;
    it = mappings_.find(&m);

    ASSERTION(it != mappings_.end(), "Mapping not registered");
    mappings_.erase(it);

    return error::DSM_SUCCESS;
}

inline
error
block::transfer_mappings(block &&b)
{
    TRACE(LOCAL, FMT_ID2" Transferring mappings from " FMT_ID2, get_print_id2(), b.get_print_id2());

    CHECK(this != &b, error::DSM_ERROR_INVALID_VALUE);
    CHECK(size_ == b.size_, error::DSM_ERROR_INVALID_VALUE);

    // TODO: check what happens the same mapping was already registered
    mappings_.insert(b.mappings_.begin(), b.mappings_.end());

    b.mappings_.clear();

    return error::DSM_SUCCESS;
}

inline
size_t
block::get_size() const
{
    return size_;
}

}}}

#endif /* GMAC_DSM_COHERENCE_BLOCK_IMPL_H_ */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
