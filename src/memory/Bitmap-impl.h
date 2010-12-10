#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

namespace __impl { namespace memory { namespace vm {

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
uint32_t Bitmap::offset(const accptr_t addr) const
{
#ifdef BITMAP_BIT
    uint32_t entry = uint32_t(addr >> (shiftPage_ + 3));
#else // BITMAP_BYTE
    uint32_t entry = uint32_t(addr >> shiftPage_);
#endif
    return entry;
}

template <bool __check, bool __clear, bool __set>
bool
Bitmap::CheckClearSet(const accptr_t addr)
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
    TRACE(LOCAL,"Bitmap check for %p -> entry %zu", (void *) addr, entry);
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
void Bitmap::set(const accptr_t addr)
{
    CheckClearSet<false, false, true>(addr);
}

inline
void Bitmap::setBlock(const accptr_t addr)
{
    unsigned subBlockSize = paramPageSize/paramBitmapChunksPerPage;
    for (unsigned i = 0; i < paramBitmapChunksPerPage; i++) {
        set(addr + i * subBlockSize);
    }
}

inline
bool Bitmap::check(const accptr_t addr)
{
    bool b = CheckClearSet<true, false, false>(addr);
    return b;
}

inline
bool Bitmap::checkBlock(const accptr_t addr)
{
    unsigned subBlockSize = paramPageSize/paramBitmapChunksPerPage;
    for (unsigned i = 0; i < paramBitmapChunksPerPage; i++) {
        if (check(addr + i * subBlockSize)) return true;
    }
    return false;
}

inline
bool Bitmap::checkAndClear(const accptr_t addr)
{
    bool b = CheckClearSet<true, true, false>(addr);
    return b;
}

inline
bool Bitmap::checkAndSet(const accptr_t addr)
{
    bool b = CheckClearSet<true, false, true>(addr);
    return b;
}

inline
void Bitmap::clear(const accptr_t addr)
{
    CheckClearSet<false, true, false>(addr);
}

inline
accptr_t Bitmap::accelerator() 
{
    if (minEntry_ != -1) {
        return accelerator_ + minEntry_;
    }

    return accelerator_;
}

inline
hostptr_t Bitmap::host() const
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
unsigned Bitmap::getSubBlock(const accptr_t addr) const
{
    return uint32_t(addr >> shiftPage_) & subBlockMask_;
}

inline
size_t Bitmap::getSubBlockSize() const
{
    return subBlockSize_;
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
void Bitmap::newRange(const accptr_t ptr, size_t count)
{
#ifdef BITMAP_BYTE
    hostptr_t start = bitmap_ + offset(ptr);
    hostptr_t end   = bitmap_ + offset(ptr + count - 1);
    ::memset(start, 0, end - start + 1);
#else // BITMAP_BIT
    hostptr_t start = bitmap_ + offset(ptr);
    hostptr_t end   = bitmap_ + offset(ptr + count - 1);
    ::memset(start, 0, end - start + 1);
#endif
    updateMaxMin(offset(ptr));
    updateMaxMin(offset(ptr + count - 1));
    TRACE(LOCAL,"ptr: %p ("FMT_SIZE")", (void *) ptr, count);
}

inline
void Bitmap::removeRange(const accptr_t ptr, size_t count)
{
    // TODO implement smarter range handling for more efficient transfers

}

}}}

#endif
