#ifndef __KERNEL_MODE_IPP
#define __KERNEL_MODE_IPP

namespace gmac {

inline void Mode::init()
{
    key.set(NULL);
}

inline bool
Mode::hasCurrent()
{
    return key.get() != NULL;
}

inline void Mode::inc()
{
    _count++;
}

inline void Mode::destroy()
{
    _count--;
    if(_count == 0) delete this;
}

inline unsigned Mode::id() const
{
    return _id;
}

inline void
Mode::addObject(memory::Object *obj)
{
    _map->insert(obj);
}

#ifndef USE_MMAP
inline void
Mode::addReplicatedObject(memory::Object *obj)
{
    _map->insertShared(obj);
}

inline void
Mode::addCentralizedObject(memory::Object *obj)
{
    _map->insertGlobal(obj);
}

#endif

inline void
Mode::removeObject(memory::Object *obj)
{
    _map->remove(obj);
}

inline memory::Object *
Mode::findObject(const void *addr)
{
    return _map->find(addr);
}

inline const memory::Map &
Mode::objects()
{
    return *_map;
}

inline gmacError_t
Mode::error() const
{
    return _error;
}

inline void
Mode::error(gmacError_t err)
{
    _error = err;
}

#ifdef USE_VM
inline memory::vm::Bitmap &
Mode::dirtyBitmap()
{
    return *_bitmap;
}

inline const memory::vm::Bitmap &
Mode::dirtyBitmap() const
{
    return *_bitmap;
}
#endif

}

#endif
