#ifndef GMAC_DSM_COHERENCE_BLOCK_IMPL_H_
#define GMAC_DSM_COHERENCE_BLOCK_IMPL_H_

namespace __impl { namespace dsm { namespace coherence {

inline
block::block(size_t size) :
    lock("block"),
    size_(size)
{
    TRACE(LOCAL, "block<"FMT_ID"> Creating "FMT_SIZE" bytes", get_print_id(), size);
}

inline
block::~block()
{
    TRACE(LOCAL, "block<"FMT_ID"> Deleting", get_print_id());
}

inline
block_ptr
block::split(size_t off)
{
    TRACE(LOCAL, "block<"FMT_ID"> Splitting "FMT_SIZE, get_print_id(), off);

    // Create new block
    block *nBlock = new block(size_ - off);
    // Set new size
    size_ = off;

    for (auto m : mappings_) {
        bool inserted = nBlock->mappings_.insert(mappings::value_type(m.first, m.second)).second;
        ASSERTION(inserted == true); 
    }

    return block_ptr(nBlock);
}

inline
void
block::shift(mapping_ptr m, size_t off)
{
    mappings::iterator it = mappings_.find(m);
    ASSERTION(it != mappings_.end());

    it->second.off_ += off;
}

inline
error
block::acquire(mapping_ptr aspace, int flags)
{
    if (flags & GMAC_PROT_WRITE) {
        lock::lock_write();
    } else if (flags & GMAC_PROT_READ) {
        lock::lock_read();
    } else {
        FATAL("Wrong GmacProtection flag for acquire");
        return DSM_ERROR_INVALID_PROT;
    }

    return DSM_SUCCESS;
}

inline
error
block::release(mapping_ptr aspace)
{
    lock::unlock();

    return DSM_SUCCESS;
}

inline
error
block::register_mapping(mapping_ptr m, size_t off)
{
    TRACE(LOCAL, "block<"FMT_ID"> Register mapping<"FMT_ID">", get_print_id(), m->get_print_id());

    ASSERTION(mappings_.find(m) == mappings_.end(), "Mapping already registered");
    mapping_descriptor descr = {
                                   off,
                                   STATE_INVALID
                               };
    mappings_.insert(mappings::value_type(m, descr));
    return DSM_SUCCESS;
}

inline
error
block::unregister_mapping(const mapping &m)
{
    TRACE(LOCAL, "block<"FMT_ID"> Unregister mapping<"FMT_ID">", get_print_id(), m.get_print_id());

    mappings::iterator it;
    it = mappings_.find(&m);

    ASSERTION(it != mappings_.end(), "Mapping not registered");
    mappings_.erase(it);

    return DSM_SUCCESS;
}

inline
error
block::transfer_mappings(block &&b)
{
    CHECK(size_ == b.size_, DSM_ERROR_INVALID_VALUE);

    TRACE(LOCAL, "block<"FMT_ID"> Transferring mappings from block<"FMT_ID">", get_print_id(), b.get_print_id());

    // TODO: check what happens the same mapping was already registered
    mappings_.insert(b.mappings_.begin(), b.mappings_.end());

    b.mappings_.clear();

    return DSM_SUCCESS;
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
