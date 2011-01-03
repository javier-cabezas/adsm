#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

#include <cmath>

namespace __impl { namespace memory { namespace vm {

TableBase::TableBase(size_t nEntries) :
    allocated_(false),
    nEntries_(nEntries),
    nUsedEntries_(0),
    firstEntry_(-1),
    lastEntry_(-1),
{
}

bool
TableBase::isAllocated()
{
    return allocated_;
}

size_t
TableBase::getNEntries() const
{
    return nEntries_;
}

size_t
TableBase::getNUsedEntries() const
{
    return nUsedEntries_;
}

long int
TableBase::firstEntry() const
{
    return firstEntry_;
}

long int
TableBase::lastEntry() const
{
    return lastEntry_;
}


template <typename T>
void
NestedBitmap<T>::alloc()
{
    assertion(entries_ == NULL);
    entries_ = new T[nEntries_];
    allocated_ = true;
}

template <typename T>
NestedBitmap<T>::NestedBitmap(size_t nEntries) :
    TableBase(nEntries), entries_(NULL), dirty_(false)
{
}

template <typename T>
T
Table<T>::getEntry(unsigned long index)
{
    assertion(allocated_ == true);

    if (level < BITMAP_LEVELS - 1) {
        return ptrs_[index]->getEntry(index);
    } else {
        return ptrs_[index];
    }
}

template <typename T>
void
Table<T>::setEntry(T value, unsigned long index)
{
    assertion(index < nEntries);
    assertion(allocated_ == true);

    return ptrs_[index] = value;
}

template <typename T>
bool
Table<T>::isDirty()
{
    return dirty_;
}

template <typename T>
inline bool
Directory<T>::exists(size_t index) const
{
    assertion(index < nEntries);
    return allocated_ && ptrs_[index] != NULL;
}

template<typename T>
void
Directory<T>::create(size_t index, size_t entries)
{
    assertion(index < nEntries_);
    assertion(ptrs_[index] == NULL);
    setEntry(new T(entries), index);
}

template<typename T>
T &
Directory<T>::getEntry(size_t index) const
{
    assertion(index < nEntries_);
    return *ptrs_[index];
}

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
uint32_t Bitmap::offset(const accptr_t addr) const
{
#ifdef BITMAP_BIT
    uint32_t entry = uint32_t((unsigned long)addr >> (shiftPage_ + 3));
#else // BITMAP_BYTE
    uint32_t entry = uint32_t((unsigned)addr >> shiftPage_);
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

inline
void Bitmap::set(const accptr_t addr)
{
    CheckClearSet<false, false, true>(addr);
}

inline
void Bitmap::setBlock(const accptr_t addr)
{
    unsigned subBlockSize = paramPageSize/paramSubBlocks;
    for (unsigned i = 0; i < paramSubBlocks; i++) {
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
    unsigned subBlockSize = paramPageSize/paramSubBlocks;
    for (unsigned i = 0; i < paramSubBlocks; i++) {
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
unsigned Bitmap::getSubBlock(const accptr_t addr) const
{
    return uint32_t((unsigned long)addr >> shiftPage_) & subBlockMask_;
}

inline
size_t Bitmap::getSubBlockSize() const
{
    return subBlockSize_;
}

inline
void Bitmap::newRange(const accptr_t ptr, size_t count)
{
#ifdef BITMAP_BYTE
    hostptr_t start = ptr;
    hostptr_t end   = ptr + count;

    unsigned startRootEntry = getRootEntry(start);
    unsigned endRootEntry   = getRootEntry(end);

    for (unsigned i = startRootEntry; i <= endRootEntry; i++) {
        // Create the memory ranges if needed
        if (bitmap_[startRootEntry] == NULL) {
            bitmap_[startRootEntry] = new EntryType[size_ * EntriesPerByte_];
            ::memset(&bitmap_[i][startRangeEntry], 0, rangeEntries_);
        }
    }
    allocations_.insert(AllocMap::value_type(ptr, count));
#else // BITMAP_BIT
    hostptr_t start = bitmap_ + offset(ptr);
    hostptr_t end   = bitmap_ + offset(ptr + count - 1);
    ::memset(start, 0, end - start + 1);
#endif
    TRACE(LOCAL,"ptr: %p ("FMT_SIZE")", (void *) ptr, count);
}

inline
void Bitmap::removeRange(const accptr_t ptr, size_t count)
{
    // TODO implement smarter range handling for more efficient transfers

}

inline
accptr_t SharedBitmap::accelerator()
{
    if (minEntry_ != -1) {
        return accelerator_ + minEntry_;
    }

    return accelerator_;
}

inline
void SharedBitmap::reset()
{
    synced_ = true;
    dirty_ = false;
}

inline
bool SharedBitmap::synced() const
{
    return synced_;
}

inline
void SharedBitmap::synced(bool s)
{
    synced_ = s;
}



}}}

#endif
