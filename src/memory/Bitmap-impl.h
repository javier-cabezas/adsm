#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

#include <cmath>

namespace __impl { namespace memory { namespace vm {

inline
size_t
Node::getNUsedEntries() const
{
    return nUsedEntries_;
}

inline
unsigned long
Node::getFirstUsedEntry() const
{
    return firstUsedEntry_;
}

inline
unsigned long
Node::getLastUsedEntry() const
{
    return lastUsedEntry_;
}

inline
void
Node::addEntries(unsigned long startIndex, unsigned long endIndex)
{
    if (nUsedEntries_ == 0) {
        firstUsedEntry_ = startIndex;
        lastUsedEntry_ = endIndex;
    } else {
        if (firstUsedEntry_ > startIndex) firstUsedEntry_ = startIndex;
        if (lastUsedEntry_ < endIndex) lastUsedEntry_ = endIndex;
    }

    nUsedEntries_ += (endIndex - startIndex + 1);

    for (unsigned long i = startIndex; i <= endIndex; i++) {
        usedEntries_[i] = true;
    }
}

inline
void
Node::removeEntries(unsigned long startIndex, unsigned long endIndex)
{
    for (unsigned long i = startIndex; i <= endIndex; i++) {
        usedEntries_[i] = false;
    }

    nUsedEntries_ -= (endIndex - startIndex + 1);

    if (nUsedEntries_ > 0) {
        bool first = false;
        for (unsigned long i = 0; i <= nEntries_; i++) {
            if (first == false && usedEntries_[i] == true) {
                firstUsedEntry_ = i;
                first = true;
            }

            if (first == true && usedEntries_[i] == true) {
                lastUsedEntry_ = i;
            }
        }
    }
}

inline
StoreHost::StoreHost(Bitmap &root, size_t size) :
    root_(root),
    entriesHost_(hostptr_t(::malloc(size))),
    size_(size)
{
}

inline
StoreHost::~StoreHost()
{
    ::free(entriesHost_);
}

inline
bool
StoreShared::isSynced() const
{
    return synced_;
}

inline
void
StoreShared::setSynced(bool synced)
{
    synced_ = synced;
}

inline
accptr_t
StoreShared::getAccAddr() const
{
    return entriesAcc_;
}

inline bool
StoreShared::isDirty() const
{
    return dirty_;
}

inline
void
StoreShared::setDirty(bool dirty)
{
    dirty_ = dirty;
}

inline
void
StoreShared::addDirtyEntry(unsigned long index)
{
    if (isDirty() == false) {
        firstDirtyEntry_ = index;
        lastDirtyEntry_ = index;
        setDirty(true);
    } else {
        if (firstDirtyEntry_ > index) firstDirtyEntry_ = index;
        if (lastDirtyEntry_ < index) lastDirtyEntry_ = index;
    }
}

inline
void
StoreShared::addDirtyEntries(unsigned long startIndex, unsigned long endIndex)
{
    if (isDirty() == false) {
        firstDirtyEntry_ = startIndex;
        lastDirtyEntry_ = endIndex;
        setDirty(true);
    } else {
        if (firstDirtyEntry_ > startIndex) firstDirtyEntry_ = startIndex;
        if (lastDirtyEntry_ < endIndex) lastDirtyEntry_ = endIndex;
    }
}

inline
StoreShared::StoreShared(Bitmap &root, size_t size) :
    StoreHost(root, size),
    entriesAcc_(NULL),
    allocatedAcc_(false),
    dirty_(false),
    synced_(true),
    firstDirtyEntry_(-1), lastDirtyEntry_(-1)
{
}

template <typename T>
inline void
StoreShared::syncToHost(unsigned long startIndex, unsigned long endIndex)
{
    syncToHost(startIndex, endIndex, sizeof(T));
}

template <typename T>
inline void
StoreShared::syncToAccelerator(unsigned long startIndex, unsigned long endIndex)
{
    syncToAccelerator(startIndex, endIndex, sizeof(T));
}

inline
unsigned long
StoreShared::getFirstDirtyEntry()
{
    return firstDirtyEntry_;
}

inline
unsigned long
StoreShared::getLastDirtyEntry()
{
    return lastDirtyEntry_;
}

inline
unsigned long
Node::getLocalIndex(unsigned long index) const
{
    return (index & mask_) >> shift_;
}

inline
unsigned long
Node::getGlobalIndex(unsigned long localIndex) const
{
    return localIndex << shift_;
}

inline
unsigned long
Node::getNextIndex(unsigned long index) const
{
    return index & ~mask_;
}

template <typename S>
NodeStore<S>::NodeStore(Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    Node(nEntries, nextEntries),
    S(root, sizeof(S) * nEntries)
{
}

template <typename S>
Node *
NodeStore<S>::getNode(unsigned long index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

template <typename S>
BitmapState
NodeStore<S>::getLeaf(unsigned long index)
{
    uint8_t val = reinterpret_cast<uint8_t *>(this->entriesHost_)[index];
    return BitmapState(val);
}

template <typename S>
Node *&
NodeStore<S>::getNodeRef(unsigned long index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

template <typename S>
uint8_t &
NodeStore<S>::getLeafRef(unsigned long index)
{
    return static_cast<uint8_t *>(this->entriesHost_)[index];
}


inline
NodeHost::NodeHost(Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    NodeStore<StoreHost>(root, nEntries, nextEntries)
{
}

inline uint8_t &
NodeShared::getAccLeafRef(unsigned long index)
{
    return static_cast<uint8_t *>((void *) this->entriesAcc_)[index];
}

inline
NodeShared::NodeShared(Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    NodeStore<StoreShared>(root, nEntries, nextEntries)
{
}

template <typename S>
BitmapState
NodeStore<S>::getEntry(unsigned long index)
{
    unsigned long localIndex = this->getLocalIndex(index);
    unsigned long nextIndex = this->getNextIndex(index);

    if (this->nextEntries_.size() == 0) {
        return getLeaf(localIndex);
    } else {
        Node *node = getNode(localIndex);
        return node->getEntry(nextIndex);
    }
}

template <typename S>
BitmapState
NodeStore<S>::getAndSetEntry(unsigned long index, BitmapState state)
{
    unsigned long localIndex = this->getLocalIndex(index);
    unsigned long nextIndex = this->getNextIndex(index);

    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        BitmapState val = BitmapState(ref);
        ref = state;
        return val;
    } else {
        Node *node = getNode(localIndex);
        return node->getAndSetEntry(nextIndex, state);
    }
}

template <typename S>
void
NodeStore<S>::setEntry(unsigned long index, BitmapState state)
{
    unsigned long localIndex = this->getLocalIndex(index);
    unsigned long nextIndex = this->getNextIndex(index);

    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        ref = state;
    } else {
        Node *node = getNode(localIndex);
        node->setEntry(nextIndex, state);
    }
}

template <typename S>
void
NodeStore<S>::setEntryRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    unsigned long localStartIndex = this->getLocalIndex(startIndex);
    unsigned long localEndIndex = this->getLocalIndex(endIndex);
 
    if (this->nextEntries_.size() == 0) {
        for (unsigned long i = localStartIndex; i <= localEndIndex; i++) {
            uint8_t &leaf = getLeafRef(i);
            leaf = state;
        }
        return;
    }

    unsigned long startWIndex = startIndex;
    unsigned long endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                this->getGlobalIndex(localStartIndex + 1) - 1;

    unsigned long i = localStartIndex;
    do {
        Node *node = getNode(i);
        node->setEntryRange(this->getNextIndex(startWIndex), this->getNextIndex(endWIndex), state);
        i++;
        startWIndex = this->getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? this->getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}

inline
BitmapState
NodeShared::getEntry(unsigned long index)
{
    sync(); 

    return NodeStore<StoreShared>::getEntry(index);
}

inline
BitmapState
NodeShared::getAndSetEntry(unsigned long index, BitmapState state)
{
    sync();

    addDirtyEntry(getLocalIndex(index));

    return NodeStore<StoreShared>::getAndSetEntry(index, state);
}

inline
void
NodeShared::setEntry(unsigned long index, BitmapState state)
{
    sync();

    addDirtyEntry(getLocalIndex(index));

    NodeStore<StoreShared>::setEntry(index, state);
}

inline
void
NodeShared::setEntryRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    sync();

    addDirtyEntries(getLocalIndex(startIndex), getLocalIndex(endIndex));

    NodeStore<StoreShared>::setEntryRange(startIndex, endIndex, state);
}

inline
void
NodeShared::sync()
{
    if (!this->isSynced()) {
        if (nextEntries_.size() > 0) {
            this->syncToHost<Node *>(getFirstUsedEntry(), getLastUsedEntry());
        } else {
            this->syncToHost<uint8_t>(getFirstUsedEntry(), getLastUsedEntry());
        }
        this->setSynced(true);
    }
}

inline
void
Bitmap::setEntry(const accptr_t addr, BitmapState state)
{
    unsigned long entry = getEntry(addr);
    root_->setEntry(entry, state);
}

inline
void
Bitmap::setEntryRange(const accptr_t addr, size_t bytes, BitmapState state)
{
    unsigned long firstEntry = getEntry(addr);
    unsigned long lastEntry = getEntry(addr + bytes - 1);
    root_->setEntryRange(firstEntry, lastEntry, state);
}

inline
unsigned long
Bitmap::getIndex(const accptr_t _ptr) const
{
    void * ptr = (void *) _ptr;
    unsigned long index = (unsigned long)ptr;
    return index >> shift_;
}

inline
BitmapState
Bitmap::getEntry(const accptr_t addr) const
{
    unsigned long entry = getEntry(addr);
    return root_->getEntry(entry);
}

inline
BitmapState
Bitmap::getAndSetEntry(const accptr_t addr, BitmapState state)
{
    unsigned long entry = getEntry(addr);
    return root_->getAndSetEntry(entry, state);
}

inline void
BitmapShared::acquire()
{
    if (synced_ == true) {
        ((NodeShared *)root_)->acquire();
        synced_ = false;
    }
}

inline void
BitmapShared::release()
{
    if (dirty_ == false) {
        // Sync the device variables
        syncToAccelerator();

        // Sync the bitmap contents
        ((NodeShared *)root_)->release();
    }
}

}}}

#endif
