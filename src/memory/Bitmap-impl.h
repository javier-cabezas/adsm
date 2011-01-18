#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

#include <cmath>

namespace __impl { namespace memory { namespace vm {

inline
Store::Store() :
    entries_(NULL),
    allocated_(false)
{
}

inline bool
StoreHost::isSynced() const
{
    return true;
}

template <typename T>
void
StoreHost::alloc(size_t nEntries)
{
    entries_ = new T[nEntries];
    allocated_ = true;
}

template <typename T>
StoreHost::StoreHost() :
    Store()
{
    entries_ = new T[nEntries];
    allocated_ = true;
}

bool
StoreShared::isSynced() const
{
    return synced_;
}

void
StoreShared::setSynced(bool synced)
{
    synced_ = synced;
}


StoreShared::StoreShared(Bitmap<StoreShared> &root) :
    Store(),
    root_(root)
{
}

template <typename S>
unsigned long
Directory<S>::getLocalIndex(unsigned long index) const
{
    return (index & mask_) >> shift_;
}

template <typename S>
unsigned long
Directory<S>::getNextIndex(unsigned long index) const
{
    return index & ~mask_;
}

template <typename S>
unsigned long
Directory<S>::getNextBaseIndex(unsigned long localIndex) const
{
    return getGlobalOffset * localIndex;
}

template <typename S>
unsigned long
Directory<S>::getGlobalOffset() const
{
    return 1 << (shift_ - 1);
}

template <typename S>
Node *
Directory<S>::get(unsigned long index)
{
    return static_cast<Node **>(this->entries_)[index];
}

template <typename S>
Node **
Directory<S>::getAddr(unsigned long index)
{
    return &static_cast<Node **>(this->entries_)[index];
}

template <typename S>
Directory<S>::Directory(Bitmap<S> &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    Node(nEntries),
    S(root),
    nextEntries_(nextEntries)
{
    unsigned shift = 0;
    for (unsigned i = 0; i < nextEntries_.size(); i++) {
        shift += unsigned(ceilf(log2f(float(nextEntries[i]))));
    }

    mask_  = (nEntries - 1) << shift;
    shift_ = shift;
}

template <typename S>
BitmapState
Directory<S>::getEntry(unsigned long index)
{
    if (!this->isSynced()) this->syncToHost();

    unsigned long localIndex = getLocalIndex(index);
    unsigned long nextIndex = getNextIndex(index);

    if (nextEntries_.size() == 1) {
        Leaf<S> *leaf = static_cast<Leaf<S> *>(get(localIndex));
        return leaf->getEntry(nextIndex);
    } else {
        Directory<S> *dir = static_cast<Directory<S> *>(get(localIndex));
        return dir->getEntry(nextIndex);
    }
}

template <typename S>
BitmapState
Directory::getAndSetEntry(unsigned long index, BitmapState state)
{
    if (!this->allocated_) this->template alloc<Node *>(nEntries_);
    if (!this->isSynced()) sync();

    unsigned long localIndex = getLocalIndex(index);
    unsigned long nextIndex = getNextIndex(index);

    if (nextEntries_.size() == 1) {
        Leaf<S> *&leaf = *reinterpret_cast<Leaf<S> **>(getAddr(localIndex));
        if (leaf == NULL) leaf = new Leaf<S> (this->root_, nextEntries_[0]);
        return leaf->setEntry(nextIndex, state);
    } else {
        Directory<S> *&dir = *reinterpret_cast<Directory<S> **>(getAddr(localIndex));
        if (dir == NULL) {
            std::vector<unsigned> nextEntries(nextEntries_.size() - 1);
            std::copy(++nextEntries_.begin(), nextEntries_.end(), nextEntries.begin());
            dir = new Directory<S>(this->root_, nextEntries_[0], nextEntries);
        }
        return dir->setEntry(nextIndex, state);
    }
}

template <typename S>
void
Directory<S>::setEntry(unsigned long index, BitmapState state)
{
    if (!this->allocated_) this->template alloc<Node *>(nEntries_);
    if (!this->isSynced()) sync();

    unsigned long localIndex = getLocalIndex(index);
    unsigned long nextIndex = getNextIndex(index);

    if (nextEntries_.size() == 1) {
        Leaf<S> *&leaf = *reinterpret_cast<Leaf<S> **>(getAddr(localIndex));
        if (leaf == NULL) leaf = new Leaf<S> (this->root_, nextEntries_[0]);
        leaf->setEntry(nextIndex, state);
    } else {
        Directory<S> *&dir = *reinterpret_cast<Directory<S> **>(getAddr(localIndex));
        if (dir == NULL) {
            std::vector<unsigned> nextEntries(nextEntries_.size() - 1);
            std::copy(++nextEntries_.begin(), nextEntries_.end(), nextEntries.begin());
            dir = new Directory<S>(this->root_, nextEntries_[0], nextEntries);
        }
        dir->setEntry(nextIndex, state);
    }
}

void setEntryRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    if (!this->allocated_) this->template alloc<Node *>(nEntries_);
    if (!this->isSynced()) sync();

    unsigned long localStartIndex = getLocalIndex(startIndex);
    unsigned long localEndIndex = getLocalIndex(endIndex);

    if (nextEntries_.size() == 1) {
        if (localStartIndex == localEndIndex) {
            Leaf<S> *&leaf = *reinterpret_cast<Leaf<S> **>(getAddr(i));
            if (leaf == NULL) leaf = new Leaf<S> (this->root_, nextEntries_[0]);
            leaf->setEntryRange(getNextIndex(localStartIndex), getNextIndex, state);
        } else {
            localStartIndex++;
            for (unsigned i = localStartIndex; i <= localEndIndex; i++) {
                unsigned long nextIndex = getNextIndex(i);
            }
        }
    } else {
        Directory<S> *&dir = *reinterpret_cast<Directory<S> **>(getAddr(localIndex));
        if (dir == NULL) {
            std::vector<unsigned> nextEntries(nextEntries_.size() - 1);
            std::copy(++nextEntries_.begin(), nextEntries_.end(), nextEntries.begin());
            dir = new Directory<S>(this->root_, nextEntries_[0], nextEntries);
        }
        dir->setEntry(nextIndex, state);
    }
}


template <typename S>
void
Directory<S>::acquire()
{
    if (nextEntries_.size() == 1) {
        Leaf<S> *leaf;
        for (unsigned i = this->firstUsedEntry_; i <= this->lastUsedEntry_; i++) {
            leaf = static_cast<Leaf<S> *>(get(i));
            if (leaf != NULL) {
                leaf->acquire();
            }
        }
    } else {
        Directory<S> *dir;
        for (unsigned i = this->firstUsedEntry_; i <= this->lastUsedEntry_; i++) {
            dir = static_cast<Directory<S> *>(get(i));
            if (dir!= NULL) {
                dir->acquire();
            }
        }
    }

    this->setSynced(false);
}

template <typename S>
void
Directory<S>::release()
{
    if (nextEntries_.size() == 1) {
        Leaf<S> *leaf;
        for (unsigned i = this->firstUsedEntry_; i <= this->lastUsedEntry_; i++) {
            leaf = static_cast<Leaf<S> *>(get(i));
            if (leaf != NULL) {
                leaf->release();
            }
        }
    } else {
        Directory<S> *dir;
        for (unsigned i = this->firstUsedEntry_; i <= this->lastUsedEntry_; i++) {
            dir = static_cast<Directory<S> *>(get(i));
            if (dir!= NULL) {
                dir->release();
            }
        }
    }

    this->syncToDevice();
    this->setSynced(true);
}

template <typename S>
BitmapState
Leaf<S>::getEntry(unsigned long index)
{
    if (!this->isSynced()) this->syncToHost();

    return BitmapState(get(index));
}

template <typename S>
void
Leaf<S>::setEntry(unsigned long index, BitmapState state)
{
    if (!this->allocated_) this->template alloc<uint8_t>(nEntries_);
    if (!this->isSynced()) this->syncToHost();

    getRef(index) = state;
}




#if 0

NestedBitmap::NestedBitmap(unsigned bits, bool shared)
{
    if (shared) {
        if (BitmapLevels > 1) {
            root = new Directory<StoreShared<Bitmap *> >(1);
        }
        else {
            root = new Directory<StoreShared<BitmapType> >(1);
        }
    else {
        if (BitmapLevels > 1) {
            root = new Directory<StoreHost<Bitmap *> >(1);
        }
        else {
            root = new Directory<StoreHost<BitmapType> >(1);
        }
    }
};


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

#endif

}}}

#endif
