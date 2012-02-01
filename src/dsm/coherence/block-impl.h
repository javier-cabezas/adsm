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
gmacError_t
block::acquire(int flags)
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
block::release()
{
    lock::unlock();

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
