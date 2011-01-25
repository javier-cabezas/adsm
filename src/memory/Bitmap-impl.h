#ifndef GMAC_MEMORY_BITMAP_H_IMPL_
#define GMAC_MEMORY_BITMAP_H_IMPL_

#include <cmath>

#include "Memory.h"

namespace __impl { namespace memory { namespace vm {

inline
unsigned
Node::getLevel() const
{
    return level_;
}

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
StoreHost::StoreHost(Bitmap &root, size_t size, bool alloc) :
    root_(root),
    size_(size)
{
    TRACE(LOCAL, "StoreHost constructor");
    if (alloc) {
        TRACE(LOCAL, "Allocating memory");
        entriesHost_ = hostptr_t(::malloc(size));
        ::memset(entriesHost_, 0, size);
    } else {
        TRACE(LOCAL, "NOT Allocating memory");
        entriesHost_ = NULL;
    }
}

inline
StoreHost::~StoreHost()
{
    TRACE(LOCAL, "StoreHost destructor");

    if (entriesHost_ != NULL) {
        ::free(entriesHost_);
    }
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
StoreShared::StoreShared(Bitmap &root, size_t size, bool allocHost) :
    StoreHost(root, size, allocHost),
    entriesAccHost_(hostptr_t(::malloc(size))),
    entriesAcc_(NULL),
    dirty_(false),
    synced_(true)
{
    TRACE(LOCAL, "StoreShared constructor");
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
NodeStore<S>::NodeStore(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    Node(level, nEntries, nextEntries),
    S(root, nEntries * (nextEntries.size() == 0? sizeof(uint8_t): sizeof(Node *)), nextEntries.size() > 0) // TODO: fix this. It does not work for StoreHost
{
    TRACE(LOCAL, "NodeStore constructor");
}

template <typename S>
Node *
NodeStore<S>::getNode(unsigned long index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

template <typename S>
Node *&
NodeStore<S>::getNodeRef(unsigned long index)
{
    return reinterpret_cast<Node **>(this->entriesHost_)[index];
}

inline
BitmapState
NodeHost::getLeaf(unsigned long index)
{
    uint8_t val = reinterpret_cast<uint8_t *>(this->entriesHost_)[index];
    return BitmapState(val);
}

inline
BitmapState
NodeShared::getLeaf(unsigned long index)
{
    uint8_t val = reinterpret_cast<uint8_t *>(this->entriesAccHost_)[index];
    return BitmapState(val);
}

inline
uint8_t &
NodeHost::getLeafRef(unsigned long index)
{
    return static_cast<uint8_t *>(this->entriesHost_)[index];
}

inline
uint8_t &
NodeShared::getLeafRef(unsigned long index)
{
    return static_cast<uint8_t *>(this->entriesAccHost_)[index];
}

inline
NodeHost::NodeHost(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    NodeStore<StoreHost>(level, root, nEntries, nextEntries)
{
    TRACE(LOCAL, "NodeHost constructor");
}

inline
NodeShared *&
NodeShared::getNodeRef(unsigned long index)
{
    return reinterpret_cast<NodeShared **>(this->entriesHost_)[index];
}

inline
NodeShared *&
NodeShared::getNodeAccHostRef(unsigned long index)
{
    return reinterpret_cast<NodeShared **>(this->entriesAccHost_)[index];
}

inline
NodeShared *
NodeShared::getNodeAccAddr(unsigned long index)
{
    return (NodeShared *) (reinterpret_cast<NodeShared **>((void *) this->entriesAcc_) + index);
}


inline
NodeShared::NodeShared(unsigned level, Bitmap &root, size_t nEntries, std::vector<unsigned> nextEntries) :
    NodeStore<StoreShared>(level, root, nEntries, nextEntries)
{
    TRACE(LOCAL, "NodeShared constructor");
}

inline
NodeShared::~NodeShared()
{
    TRACE(LOCAL, "NodeShared destructor");
    freeAcc(getLevel() == 0);
}


template <typename S>
BitmapState
NodeStore<S>::getEntry(unsigned long index)
{
    unsigned long localIndex = this->getLocalIndex(index);

    TRACE(LOCAL, "getEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        return getLeaf(localIndex);
    } else {
        unsigned long nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        return node->getEntry(nextIndex);
    }
}

template <typename S>
BitmapState
NodeStore<S>::getAndSetEntry(unsigned long index, BitmapState state)
{
    unsigned long localIndex = this->getLocalIndex(index);

    TRACE(LOCAL, "getAndSetEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        BitmapState val = BitmapState(ref);
        ref = state;
        return val;
    } else {
        unsigned long nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        return node->getAndSetEntry(nextIndex, state);
    }
}

template <typename S>
void
NodeStore<S>::setEntry(unsigned long index, BitmapState state)
{
    unsigned long localIndex = this->getLocalIndex(index);

    TRACE(LOCAL, "setEntry 0x%lx", localIndex);
    if (this->nextEntries_.size() == 0) {
        uint8_t &ref = getLeafRef(localIndex);
        ref = state;
    } else {
        unsigned long nextIndex = this->getNextIndex(index);
        Node *node = getNode(localIndex);
        ASSERTION(node != NULL);
        node->setEntry(nextIndex, state);
    }
}

template <typename S>
void
NodeStore<S>::setEntryRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    unsigned long localStartIndex = this->getLocalIndex(startIndex);
    unsigned long localEndIndex = this->getLocalIndex(endIndex);
 
    TRACE(LOCAL, "setEntryRange 0x%lx 0x%lx", localStartIndex, localEndIndex);
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

template <typename S>
bool
NodeStore<S>::isAnyInRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    unsigned long localStartIndex = this->getLocalIndex(startIndex);
    unsigned long localEndIndex = this->getLocalIndex(endIndex);
 
    TRACE(LOCAL, "isAnyInRange 0x%lx 0x%lx", localStartIndex, localEndIndex);
    if (this->nextEntries_.size() == 0) {
        for (unsigned long i = localStartIndex; i <= localEndIndex; i++) {
            if (getLeaf(i) == state) return true;
        }
    }

    unsigned long startWIndex = startIndex;
    unsigned long endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                this->getGlobalIndex(localStartIndex + 1) - 1;

    unsigned long i = localStartIndex;
    do {
        Node *node = getNode(i);
        bool ret = node->isAnyInRange(this->getNextIndex(startWIndex), this->getNextIndex(endWIndex), state);
        if (ret == true) return true;
        i++;
        startWIndex = this->getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? this->getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);

    return false;
}


inline
BitmapState
NodeShared::getEntry(unsigned long index)
{
    sync(); 

    TRACE(LOCAL, "getEntry 0x%lx", getLocalIndex(index));

    return NodeStore<StoreShared>::getEntry(index);
}

inline
BitmapState
NodeShared::getAndSetEntry(unsigned long index, BitmapState state)
{
    sync();

    unsigned long localIndex = getLocalIndex(index);
    TRACE(LOCAL, "getAndSetEntry 0x%lx", localIndex);
    addDirtyEntry(localIndex);

    return NodeStore<StoreShared>::getAndSetEntry(index, state);
}

inline
void
NodeShared::setEntry(unsigned long index, BitmapState state)
{
    sync();

    unsigned long localIndex = getLocalIndex(index);
    TRACE(LOCAL, "setEntry 0x%lx", localIndex);
    addDirtyEntry(localIndex);

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
bool
NodeShared::isAnyInRange(unsigned long startIndex, unsigned long endIndex, BitmapState state)
{
    sync();

    TRACE(LOCAL, "isAnyInRange 0x%lx 0x%lx", getLocalIndex(startIndex), getLocalIndex(endIndex));

    return NodeStore<StoreShared>::isAnyInRange(startIndex, endIndex, state);
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
    TRACE(LOCAL, "setEntry %p", (void *) addr);

    unsigned long entry = getIndex(addr);
    root_->setEntry(entry, state);
}

inline
void
Bitmap::setEntryRange(const accptr_t addr, size_t bytes, BitmapState state)
{
    TRACE(LOCAL, "setEntryRange %p %zd", (void *) addr, bytes);

    unsigned long firstEntry = getIndex(addr);
    unsigned long lastEntry = getIndex(addr + bytes - 1);
    root_->setEntryRange(firstEntry, lastEntry, state);
}

inline
unsigned long
Bitmap::getIndex(const accptr_t _ptr) const
{
    void * ptr = (void *) _ptr;
    unsigned long index = (unsigned long)ptr;
    index >>= SubBlockShift_;
    return index;
}

inline
BitmapState
Bitmap::getEntry(const accptr_t addr) const
{

    unsigned long entry = getIndex(addr);
    BitmapState state = root_->getEntry(entry);

    TRACE(LOCAL, "getEntry %p: %d", (void *) addr, state);
    return state;
}

inline
BitmapState
Bitmap::getAndSetEntry(const accptr_t addr, BitmapState state)
{
    unsigned long entry = getIndex(addr);
    BitmapState ret= root_->getAndSetEntry(entry, state);
    TRACE(LOCAL, "getAndSetEntry %p: %d", (void *) addr, ret);
    return ret;
}


inline
bool
Bitmap::isAnyInRange(const accptr_t addr, size_t size, BitmapState state)
{
    TRACE(LOCAL, "isAnyInRange %p %zd", (void *) addr, size);

    unsigned long firstEntry = getIndex(addr);
    unsigned long lastEntry = getIndex(addr + size - 1);
    return root_->isAnyInRange(firstEntry, lastEntry, state);
}

inline void
BitmapShared::acquire()
{
    TRACE(LOCAL, "Acquire");

    if (synced_ == true) {
        TRACE(LOCAL, "Acquiring");
        ((NodeShared *)root_)->acquire();
        synced_ = false;
    }
}

inline void
BitmapShared::release()
{
    TRACE(LOCAL, "Release");

    if (dirty_ == false) {
        TRACE(LOCAL, "Releasing");
        // Sync the device variables
        syncToAccelerator();

        // Sync the bitmap contents
        ((NodeShared *)root_)->release();
    }
}

}}}

#endif
