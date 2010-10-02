#ifndef GMAC_MEMORY_BITMAP_IPP_
#define GMAC_MEMORY_BITMAP_IPP_

namespace gmac { namespace memory { namespace vm {

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
off_t Bitmap::offset(const void *addr) const
{
#ifdef BITMAP_BIT
    off_t entry = to32bit(addr) >> (shiftEntry_ + 5);
#else
    off_t entry = to32bit(addr) >> shiftPage_;
#endif
    return entry;
}

template <bool __check, bool __clear, bool __set>
inline
bool Bitmap::CheckClearSet(const void * addr)
{
    if (!_synced) {
        syncHost();
    }
    bool ret = false;
    size_t entry = offset(addr);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
    if (__clear || __set) {
        trace("Bitmap entry before: 0x%x", bitmap_[entry]);
    }
#ifdef BITMAP_BIT
    uint32_t val = 1 << ((to32bit(addr) >> shiftEntry_) & bitMask_);
    if(__check && (bitmap_[entry] & val) != 0) ret = true;
    if(__clear) {
        bitmap_[entry] &= ~val;
        dirty_ = true;
    }
#else
    if(__check && bitmap_[entry] != 0) {
        ret = true;
    }
    if (__clear) {
        bitmap_[entry] = 0;
        dirty_ = true;
    }
    if (__set) {
        bitmap_[entry] = 2;
        dirty_ = true;
    }
#endif
    if (__clear || __set) {
        trace("Bitmap entry after: 0x%x", bitmap_[entry]);
    } else {
        trace("Bitmap entry: 0x%x", bitmap_[entry]);
    }
    return ret;
}

inline
bool Bitmap::check(const void *addr)
{
    bool b = CheckClearSet<true, false, false>(addr);
    return b;
}


inline
bool Bitmap::checkAndClear(const void *addr)
{
    bool b = CheckClearSet<true, true, false>(addr);
    return b;
}

inline
bool Bitmap::checkAndSet(const void *addr)
{
    bool b = CheckClearSet<true, false, true>(addr);
    return b;
}

inline
void Bitmap::clear(const void *addr)
{
    CheckClearSet<false, true, false>(addr);
}

inline
void Bitmap::set(const void *addr)
{
    CheckClearSet<false, false, true>(addr);
}

inline
void *Bitmap::device() 
{
    if (device_ == NULL) allocate();
    if (minAddr_ != NULL) {
        assertion(maxAddr_ != NULL && maxAddr_ > minAddr_);
        return ((uint8_t *)device_) + offset(minAddr_);
    } else {
        return device_;
    }
}

inline
void *Bitmap::deviceBase() 
{
    if (device_ == NULL) allocate();
    return device_;
}

inline
void *Bitmap::host() const
{
    if (minAddr_ != NULL) {
        assertion(maxAddr_ != NULL && maxAddr_ > minAddr_);
        return ((uint8_t *)bitmap_) + offset(minAddr_);
    } else {
        return bitmap_;
    }
}

inline
const size_t Bitmap::size() const
{
    if (minAddr_ != NULL) {
        assertion(maxAddr_ != NULL && maxAddr_ > minAddr_);
        return (offset(maxAddr_) - offset(minAddr_) + 1) * sizeof(bitmap_[0]);
    } else {
        return size_;
    }
}

inline
const size_t Bitmap::shiftPage() const
{
    return shiftPage_;
}

#ifdef BITMAP_BIT
inline
const size_t Bitmap::shiftEntry() const
{
    return shiftEntry_;
}
#endif

inline
bool Bitmap::clean() const
{
    return !_dirty;
}


inline
void Bitmap::reset()
{
    synced_ = true;
    dirty_ = false;
}


inline
bool Bitmap::synced() const
{
    return synced_;
}

inline
void Bitmap::synced(bool s)
{
    synced_ = s;
}

inline
void Bitmap::newRange(const void * ptr, size_t count)
{
    if (device_ == NULL) allocate();
    if (minAddr_ == NULL) {
        minAddr_ = ptr;
        maxAddr_ = ((uint8_t *) ptr) + count - 1;
    } else if (ptr < minAddr_) {
        minAddr_ = ptr;
    } else if (((uint8_t *) ptr) + count > maxAddr_) {
        maxAddr_ = ((uint8_t *) ptr) + count - 1;
    }
    trace("ptr: %p (%zd) minAddr: %p maxAddr: %p", ptr, count, minAddr_, maxAddr_);
}

inline
void Bitmap::removeRange(const void * ptr, size_t count)
{
    if (ptr == minAddr_) {
        if (((uint8_t *) ptr) + count == maxAddr_) {
            maxAddr_ = NULL;
            minAddr_ = NULL;
        } else {
            minAddr_ = ((uint8_t *) ptr) + count;
        }
    } else if (((uint8_t *) ptr) + count == maxAddr_) {
        maxAddr_ = ptr;
    }
    trace("minAddr: %p maxAddr: %p", minAddr_, maxAddr_);
}

}}}

#endif
