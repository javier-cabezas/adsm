#ifndef GMAC_DSM_COHERENCE_BLOCK_IMPL_H_
#define GMAC_DSM_COHERENCE_BLOCK_IMPL_H_

namespace __impl { namespace dsm { namespace coherence {

inline
block::block(size_t size) :
    lock("block"),
    size_(size)
{
    TRACE(LOCAL, "Created new block %p", this);
}

inline
block::~block()
{
    TRACE(LOCAL, "Deleting block %p", this);
}

inline
block_ptr
block::split(size_t off)
{
    // Create new block
    block *b = new block(size_ - off);
    // Set new size
    size_ = off;

    for (mappings::iterator it  = mappings_.begin();
                            it != mappings_.end();
                          ++it) {
        bool inserted = b->mappings_.insert(mappings::value_type(it->first, it->second)).second;
        ASSERTION(inserted == true); 
    }

    return block_ptr(b);
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
    TRACE(LOCAL, "Register mapping into block %p", this);

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
block::unregister_mapping(mapping_ptr m)
{
    TRACE(LOCAL, "Unregister mapping from block %p", this);

    mappings::iterator it;
    it = mappings_.find(m);

    ASSERTION(it != mappings_.end(), "Mapping not registered");
    mappings_.erase(it);

    return DSM_SUCCESS;
}

inline
error
block::transfer_mappings(block_ptr b)
{
    TRACE(LOCAL, "Transferring mappings from block %p -> %p", b.get(), this);

    CHECK(size_ == b->size_, DSM_ERROR_INVALID_VALUE);

    for (mappings::iterator it  = b->mappings_.begin();
                            it != b->mappings_.end();
                          ++it) {
        ASSERTION(mappings_.find(it->first) == mappings_.end(), "Mapping already registered in this block"); 
        bool inserted = mappings_.insert(mappings::value_type(it->first, it->second)).second;
        ASSERTION(inserted == true); 
    }

    b.reset();

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
