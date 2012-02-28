#ifndef GMAC_DSM_COHERENCE_BLOCK_IMPL_H_
#define GMAC_DSM_COHERENCE_BLOCK_IMPL_H_

namespace __impl { namespace dsm { namespace coherence {

inline
block::block(size_t size) :
    lock("block"),
    size_(size)
{
}

inline
block_ptr
block::split(size_t off)
{
    block *b = new block(size_ - off);
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
gmacError_t
block::acquire(mapping_ptr aspace, int flags)
{
    if (flags & GMAC_PROT_WRITE) {
        lock::lock_write();
    } else if (flags & GMAC_PROT_READ) {
        lock::lock_read();
    } else {
        FATAL("Wrong GmacProtection flag for acquire");
        return gmacErrorUnknown;
    }

    return gmacSuccess;
}

inline
gmacError_t
block::release(mapping_ptr aspace)
{
    lock::unlock();

    return gmacSuccess;
}

inline
gmacError_t
block::register_mapping(mapping_ptr m, size_t off)
{
    ASSERTION(mappings_.find(m) == mappings_.end(), "Mapping already registered");
    mapping_descriptor descr = {
                                   off,
                                   STATE_INVALID
                               };
    mappings_.insert(mappings::value_type(m, descr));
    return gmacSuccess;
}

inline
gmacError_t
block::unregister_mapping(mapping_ptr m)
{
    mappings::iterator it;
    it = mappings_.find(m);

    ASSERTION(it != mappings_.end(), "Mapping not registered");
    mappings_.erase(it);

    return gmacSuccess;
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
