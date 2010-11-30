#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

namespace __impl { namespace memory { namespace vm {

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
unsigned Bitmap::offset(const void *addr) const
{
#ifdef BITMAP_BIT
    off_t entry = to32bit(addr) >> (shiftPage_ + 3);
#else // BITMAP_BYTE
    off_t entry = to32bit(addr) >> shiftPage_;
#endif
    return entry;
}

template <bool __check, bool __clear, bool __set>
bool
Bitmap::CheckClearSet(const void * addr)
{
    bool ret = false;
    size_t entry = offset(addr);
    
#ifdef BITMAP_BIT
    uint8_t val = 1 << ((to32bit(addr) >> shiftPage_) & bitMask_);
    if (__check && (bitmap_[entry] & val) != 0) ret = true;
#else // BITMAP_BYTE
    if (__check && bitmap_[entry] != 0) ret = true;
#endif

#ifdef BITMAP_BIT
    TRACE(LOCAL,"Bitmap check for %p -> entry %zu, offset %zu", addr, entry, (to32bit(addr) >> shiftPage_) & bitMask_);
#else // BITMAP_BYTE
    TRACE(LOCAL,"Bitmap check for %p -> entry %zu", addr, entry);
#endif
    if (__clear || __set) {
        TRACE(LOCAL,"Bitmap entry before: 0x%x", bitmap_[entry]);
    }

    if (__clear) {
#ifdef BITMAP_BIT
        bitmap_[entry] &= ~val;
#else // BITMAP_BYTE
        bitmap_[entry] = 0;
#endif
        dirty_ = true;
    }
    if (__set) {
#ifdef BITMAP_BIT
        FATAL(0, "Operation not supported by bit implementation");
#endif
        bitmap_[entry] = 2;
        dirty_ = true;
    }
    if (__clear || __set) {
        TRACE(LOCAL,"Bitmap entry after: 0x%x", bitmap_[entry]);
    } else {
        TRACE(LOCAL,"Bitmap entry: 0x%x", bitmap_[entry]);
    }
    return ret;
}

inline void
Bitmap::updateMaxMin(unsigned entry)
{
    if (minEntry_ == -1) {
        minEntry_ = entry;
        maxEntry_ = entry;
    } else if (entry < unsigned(minEntry_)) {
        minEntry_ = entry;
    } else if (entry > unsigned(maxEntry_)) {
        maxEntry_ = entry;
    }
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
void *Bitmap::accelerator() 
{
    if (accelerator_ == NULL) allocate();
    if (minEntry_ != -1) {
        return accelerator_ + minEntry_;
    }

    return accelerator_;
}

inline
void *Bitmap::host() const
{
    if (minEntry_ != -1) {
        return bitmap_ + minEntry_;
    } else {
        return bitmap_;
    }
}

inline
const size_t Bitmap::size() const
{
    if (minEntry_ != -1) {
        return maxEntry_ - minEntry_ + 1;
    } else {
        return 0;
    }
}

inline
bool Bitmap::clean() const
{
    return !dirty_;
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
#ifdef USE_HOSTMAP_VM
    if (accelerator_ == NULL) allocate();
#endif

#ifdef BITMAP_BYTE
    uint8_t *start = bitmap_ + offset(ptr);
    uint8_t *end   = bitmap_ + offset(static_cast<const uint8_t *>(ptr) + count - 1);
    ::memset(start, 0, end - start + 1);
#else // BITMAP_BIT
    uint8_t *start = bitmap_ + offset(ptr);
    uint8_t *end   = bitmap_ + offset(static_cast<const uint8_t *>(ptr) + count - 1);
    ::memset(start, 0, end - start + 1);
#endif
    updateMaxMin(offset(ptr));
    updateMaxMin(offset(static_cast<const uint8_t *>(ptr) + count - 1));
    TRACE(LOCAL,"ptr: %p ("FMT_SIZE")", ptr, count);
}

inline
void Bitmap::removeRange(const void * ptr, size_t count)
{
    // TODO implement smarter range handling for more efficient transfers

}

}}}

#endif
