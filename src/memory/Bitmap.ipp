#ifndef __MEMORY_BITMAP_IPP
#define __MEMORY_BITMAP_IPP

namespace gmac { namespace memory { namespace vm {

#define to32bit(a) ((unsigned long)a & 0xffffffff)

inline
off_t Bitmap::offset(const void *addr) const
{
#ifdef BITMAP_BIT
    off_t entry = to32bit(addr) >> (_shiftEntry + 5);
#else
    off_t entry = to32bit(addr) >> _shiftEntry;
#endif
    return entry;
}

template <bool __check, bool __clear>
inline
bool Bitmap::CheckAndClear(const void * addr)
{
    if (!_synced) {
        sync();
    }
    bool ret = false;
    size_t entry = offset(addr);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
    trace("Bitmap entry before: 0x%x", _bitmap[entry]);
#ifdef BITMAP_BIT
    uint32_t val = 1 << ((to32bit(addr) >> _shiftEntry) & _bitMask);
    if(__check && (_bitmap[entry] & val) != 0) ret = true;
    if(__clear) {
        _bitmap[entry] &= ~val;
        _dirty = true;
    }
#else
    if (_shiftEntry != _shiftPage) {
        uint32_t chunkIdx = entry & _bitMask;
        entry += chunkIdx;
    }
    if(__check && _bitmap[entry] != 0) {
        ret = true;
    }
    if (__clear) {
        _bitmap[entry] = 0;
        _dirty = true;
    }
#endif
    trace("Bitmap entry after: 0x%x", _bitmap[entry]);
    return ret;
}

inline
bool Bitmap::check(const void *addr)
{
    bool b = CheckAndClear<true, false>(addr);
    return b;
}


inline
bool Bitmap::checkAndClear(const void *addr)
{
    bool b = CheckAndClear<true, true>(addr);
    return b;
}

inline
void Bitmap::clear(const void *addr)
{
    CheckAndClear<false, true>(addr);
}

inline
void *Bitmap::device() 
{
    if (_device == NULL) allocate();
    return _device;
}

inline
void *Bitmap::device(const void * addr) 
{
    if (_device == NULL) allocate();
    return ((uint8_t *)_device) + offset(addr);
}

inline
void *Bitmap::host() const
{
    return _bitmap;
}

inline
void *Bitmap::host(const void * addr) const
{
    return ((uint8_t *)_bitmap) + offset(addr);
}

inline
const size_t Bitmap::size() const
{
    return _size;
}

inline
const size_t Bitmap::size(const void * start, size_t size) const
{
    return offset(((uint8_t *) start) + size) - offset(start);
}

inline
const size_t Bitmap::shiftPage() const
{
    return _shiftPage;
}

inline
const size_t Bitmap::shiftEntry() const
{
    return _shiftEntry;
}

inline
bool Bitmap::clean() const
{
    return !_dirty;
}

inline
void Bitmap::reset()
{
    _dirty = false;
}


inline
bool Bitmap::synced() const
{
    return _synced;
}

inline
void Bitmap::synced(bool s)
{
    _synced = s;
}

}}}

#endif
