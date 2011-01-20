#ifdef USE_VM

#include <cstring>

#include "core/Mode.h"

#include "Bitmap.h"

namespace __impl { namespace memory { namespace vm {

const unsigned &Bitmap::BitmapLevels_ = paramBitmapLevels;
const unsigned &Bitmap::L1Entries_ = paramBitmapL1Entries;
const unsigned &Bitmap::L2Entries_ = paramBitmapL2Entries;
const unsigned &Bitmap::L3Entries_ = paramBitmapL3Entries;
const size_t &Bitmap::BlockSize_ = paramBlockSize;
const unsigned &Bitmap::SubBlocks_ = paramSubBlocks;

Node::Node(size_t nEntries, std::vector<unsigned> nextEntries) :
    nEntries_(0), nUsedEntries_(0),
    usedEntries_(nEntries),
    firstUsedEntry_(-1), lastUsedEntry_(-1),
    nextEntries_(nextEntries)
{
    unsigned shift = 0;
    for (size_t i = 0; i < nextEntries_.size(); i++) {
        shift += unsigned(ceilf(log2f(float(nextEntries[i]))));
    }

    for (size_t i = 0; i < nEntries; i++) {
        usedEntries_[i] = false;
    }

    mask_  = (nEntries - 1) << shift;
    shift_ = shift;
}

NodeHost::~NodeHost()
{
    if (nextEntries_.size() > 0) {
        for (unsigned long i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            Node *node = getNode(i);
            if (node != NULL) {
                delete node;
            }
        }
    }
}

template <typename S>
template <typename T>
void
NodeStore<S>::registerRange(Bitmap &root, unsigned long startIndex, unsigned long endIndex)
{
    unsigned long localStartIndex = getLocalIndex(startIndex);
    unsigned long localEndIndex = getLocalIndex(endIndex);

    addEntries(localStartIndex, localEndIndex);

    if (nextEntries_.size() == 0) {
        for (unsigned long i = localStartIndex; i <= localEndIndex; i++) {
            uint8_t &leaf = getLeafRef(i);
            leaf = uint8_t(BITMAP_UNSET);
        }
        return;
    }

    unsigned long startWIndex = startIndex;
    unsigned long endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                getGlobalIndex(localStartIndex + 1) - 1;

    unsigned long i = localStartIndex;
    do {
        Node *&node = getNodeRef(i);
        std::vector<unsigned> nextEntries;
        std::copy(++nextEntries_.begin(), nextEntries_.end(), nextEntries.begin());
        node = new T(root, nextEntries_[0], nextEntries);
        node->registerRange(getNextIndex(startWIndex), getNextIndex(endWIndex));
        i++;
        startWIndex = getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}

template <typename S>
void
NodeStore<S>::unregisterRange(unsigned long startIndex, unsigned long endIndex)
{
    unsigned long localStartIndex = getLocalIndex(startIndex);
    unsigned long localEndIndex = getLocalIndex(endIndex);

    if (nextEntries_.size() == 0) {
        removeEntries(localStartIndex, localEndIndex);

        for (unsigned long i = localStartIndex; i <= localEndIndex; i++) {
            uint8_t &leaf = getLeafRef(i);
            leaf = uint8_t(BITMAP_UNSET);
        }
        return;
    }

    unsigned long startWIndex = startIndex;
    unsigned long endWIndex   = (localStartIndex == localEndIndex)? endIndex:
                                getGlobalIndex(localStartIndex + 1) - 1;

    unsigned long i = localStartIndex;
    do {
        Node *&node = getNodeRef(i);
        node->registerRange(getNextIndex(startWIndex), getNextIndex(endWIndex));
        if (node->getNUsedEntries() == 0) {
            delete node;
            node = NULL;
        }
        i++;
        startWIndex = getGlobalIndex(i);
        endWIndex = (i < localEndIndex)? getGlobalIndex(i + 1) - 1: endIndex;
    } while (i <= localEndIndex);
}


void
NodeHost::registerRange(unsigned long startIndex, unsigned long endIndex)
{
    NodeStore<StoreHost>::registerRange<NodeHost>(root_, startIndex, endIndex);
}

void
NodeShared::registerRange(unsigned long startIndex, unsigned long endIndex)
{
    NodeStore<StoreShared>::registerRange<NodeShared>(root_, startIndex, endIndex);
}

NodeShared::~NodeShared()
{
    if (nextEntries_.size() > 0) {
        for (unsigned long i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            Node *node = getNode(i);
            if (node != NULL) {
                delete node;
            }
        }
    }
}

void
NodeShared::acquire()
{
    if (nextEntries_.size() > 0) {
        for (unsigned long i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            NodeShared *node = (NodeShared *) getNode(i);
            if (node != NULL) node->acquire();
        }
    } else {
        // Only leaf nodes must be synced
        this->setSynced(false);
    }
}

void
NodeShared::release()
{
    if (nextEntries_.size() > 0) {
        for (unsigned long i = getFirstUsedEntry(); i <= getLastUsedEntry(); i++) {
            NodeShared *node = (NodeShared *) getNode(i);
            if (node != NULL) node->release();
        }
    }

    if (this->isDirty()) {
        if (nextEntries_.size() > 0) {
            this->syncToAccelerator<Node *>(this->firstDirtyEntry_, this->lastDirtyEntry_);
        } else {
            this->syncToAccelerator<uint8_t>(this->firstDirtyEntry_, this->lastDirtyEntry_);
        }
    }
}

Bitmap::Bitmap(core::Mode &mode, bool shared) :
    mode_(mode)
{
}

Bitmap::~Bitmap()
{
}

void
Bitmap::cleanUp()
{
    delete root_;
}

BitmapHost::BitmapHost(core::Mode &mode) :
    Bitmap(mode, false)
{
    std::vector<unsigned> nextEntries(BitmapLevels_ - 1);

    if (BitmapLevels_ > 1) {
        nextEntries.push_back(L2Entries_);
    }
    if (BitmapLevels_ == 3) {
        nextEntries.push_back(L3Entries_);
    }

    shift_ = unsigned(ceilf(log2f(float()))); 

    if (BitmapLevels_ > 1) {
        root_ = new NodeHost(*this, L1Entries_, nextEntries);
    }
}

BitmapShared::BitmapShared(core::Mode &mode) :
    Bitmap(mode, true),
    synced_(true)
{
    std::vector<unsigned> nextEntries(BitmapLevels_ - 1);

    if (BitmapLevels_ > 1) {
        nextEntries.push_back(L2Entries_);
    }
    if (BitmapLevels_ == 3) {
        nextEntries.push_back(L3Entries_);
    }

    shift_ = unsigned(ceilf(log2f(float()))); 

    if (BitmapLevels_ > 1) {
        root_ = new NodeShared(*this, L1Entries_, nextEntries);
    }
}

void
Bitmap::registerRange(const accptr_t addr, size_t bytes)
{
    root_->registerRange(getEntry(addr), getEntry(addr + bytes - 1));
}

void
Bitmap::unregisterRange(const accptr_t addr, size_t bytes)
{
    root_->unregisterRange(getEntry(addr), getEntry(addr + bytes - 1));
}

#if 0
#ifdef BITMAP_BIT
const unsigned Bitmap::EntriesPerByte_ = 8;
#else // BITMAP_BYTE
const unsigned Bitmap::EntriesPerByte_ = 1;
#endif

Bitmap::Bitmap(core::Mode &mode, unsigned bits) :
    RWLock("Bitmap"), bits_(bits), mode_(mode), bitmap_(NULL), dirty_(true), minPtr_(NULL), maxPtr_(NULL)
{
    unsigned rootEntries = (1 << bits) >> 32;
    if (rootEntries == 0) rootEntries = 1;
    rootEntries_ = rootEntries;

    bitmap_ = new hostptr_t[rootEntries];
    ::memset(bitmap_, 0, rootEntries * sizeof(hostptr_t));

    shiftBlock_ = int(log2(paramPageSize));
    shiftPage_  = shiftBlock_ - int(log2(paramSubBlocks));

    subBlockSize_ = (paramSubBlocks) - 1;
    subBlockMask_ = (paramSubBlocks) - 1;
    pageMask_     = subBlockSize_ - 1;

    size_    = (1 << (bits - shiftPage_)) / EntriesPerByte_;
#ifdef BITMAP_BIT
    bitMask_ = (1 << 3) - 1;
#endif

    TRACE(LOCAL, "Pages: %u", 1 << (bits - shiftPage_));
    TRACE(LOCAL,"Size : %u", size_);
}

Bitmap::Bitmap(const Bitmap &base) :
    RWLock("Bitmap"),
    bits_(base.bits_),
    mode_(base.mode_),
    bitmap_(base.bitmap_),
    dirty_(true),
    shiftBlock_(base.shiftBlock_),
    shiftPage_(base.shiftPage_),
    subBlockSize_(base.subBlockSize_),
    subBlockMask_(base.subBlockMask_),
    pageMask_(base.pageMask_),
#ifdef BITMAP_BIT
    bitMask_(base.bitMask_),
#endif
    size_(base.size_),
    minEntry_(-1), maxEntry_(-1)
{
}


Bitmap::~Bitmap()
{
    
}

void
Bitmap::cleanUp()
{
    for (long int i = minRootEntry_; i <= maxRootEntry_; i++) {
        if (bitmap_[i] != NULL) {
            delete [] bitmap_[i];
        }
    }
    delete [] bitmap_;
}

SharedBitmap::SharedBitmap(core::Mode &mode, unsigned bits) :
    Bitmap(mode, bits), linked_(false), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::SharedBitmap(const Bitmap &host) :
    Bitmap(host), linked_(true), synced_(true), accelerator_(NULL)
{
}

SharedBitmap::~SharedBitmap()
{
}


#ifdef DEBUG_BITMAP
void Bitmap:dump()
{
    core::Context * ctx = Mode::current()->context();
    ctx->invalidate();

    static int idx = 0;
    char path[256];
    sprintf(path, "_bitmap__%d", idx++);
    FILE * file = fopen(path, "w");
    fwrite(bitmap_, 1, size_, file);    
    fclose(file);
}
#endif
#endif

}}}

#endif
