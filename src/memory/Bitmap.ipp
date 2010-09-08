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
    off_t entry = to32bit(addr) >> _shiftPage;
#endif
    return entry;
}

template <bool __check, bool __clear, bool __set>
inline
bool Bitmap::CheckClearSet(const void * addr)
{
    if (!_synced) {
        sync();
    }
    bool ret = false;
    size_t entry = offset(addr);
    trace("Bitmap check for %p -> entry %zu", addr, entry);
    if (__clear || __set) {
        trace("Bitmap entry before: 0x%x", _bitmap[entry]);
    }
#ifdef BITMAP_BIT
    uint32_t val = 1 << ((to32bit(addr) >> _shiftEntry) & _bitMask);
    if(__check && (_bitmap[entry] & val) != 0) ret = true;
    if(__clear) {
        _bitmap[entry] &= ~val;
        _dirty = true;
    }
#else
    if(__check && _bitmap[entry] != 0) {
        ret = true;
    }
    if (__clear) {
        _bitmap[entry] = 0;
        _dirty = true;
    }
    if (__set) {
        _bitmap[entry] = 2;
        _dirty = true;
    }
#endif
    if (__clear || __set) {
        trace("Bitmap entry after: 0x%x", _bitmap[entry]);
    } else {
        trace("Bitmap entry: 0x%x", _bitmap[entry]);
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
    if (_device == NULL) allocate();
    if (_minAddr != NULL) {
        assertion(_maxAddr != NULL && _maxAddr > _minAddr);
        return ((uint8_t *)_device) + offset(_minAddr);
    } else {
        return _device;
    }
}

inline
void *Bitmap::deviceBase() 
{
    if (_device == NULL) allocate();
    return _device;
}

inline
void *Bitmap::host() const
{
    if (_minAddr != NULL) {
        assertion(_maxAddr != NULL && _maxAddr > _minAddr);
        return ((uint8_t *)_bitmap) + offset(_minAddr);
    } else {
        return _bitmap;
    }
}

inline
const size_t Bitmap::size() const
{
    if (_minAddr != NULL) {
        assertion(_maxAddr != NULL && _maxAddr > _minAddr);
        return (offset(_maxAddr) - offset(_minAddr)) * sizeof(_bitmap[0]);
    } else {
        return _size;
    }
}

inline
const size_t Bitmap::shiftPage() const
{
    return _shiftPage;
}

#ifdef BITMAP_BIT
inline
const size_t Bitmap::shiftEntry() const
{
    return _shiftEntry;
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

inline
void Bitmap::newRange(const void * ptr, size_t count)
{
    if (_device == NULL) allocate();
    if (_minAddr == NULL) {
        _minAddr = ptr;
        _maxAddr = ((uint8_t *) ptr) + count - 1;
    } else if (ptr < _minAddr) {
        _minAddr = ptr;
    } else if (((uint8_t *) ptr) + count > _maxAddr) {
        _maxAddr = ((uint8_t *) ptr) + count - 1;
    }
    trace("ptr: %p (%zd) minAddr: %p maxAddr: %p", ptr, count, _minAddr, _maxAddr);
}

inline
void Bitmap::removeRange(const void * ptr, size_t count)
{
    if (ptr == _minAddr) {
        if (((uint8_t *) ptr) + count == _maxAddr) {
            _maxAddr = NULL;
            _minAddr = NULL;
        } else {
            _minAddr = ((uint8_t *) ptr) + count;
        }
    } else if (((uint8_t *) ptr) + count == _maxAddr) {
        _maxAddr = ptr;
    }
    trace("minAddr: %p maxAddr: %p", _minAddr, _maxAddr);
}

}}}

#endif
