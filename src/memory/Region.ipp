#ifndef __REGION_IPP_
#define __REGION_IPP_

inline addr_t
Region::__addr(void *addr) const
{
    return (addr_t)addr;
}

inline addr_t
Region::__addr(const void *addr) const
{
    return (addr_t)addr;
}

inline void *
Region::__void(addr_t addr) const
{
    return (void *)addr;
}

inline Context *
Region::owner()
{
    return _context;
}

inline size_t
Region::size() const
{
    return _size;
}

inline void *
Region::start() const
{
    return __void(_addr);
}

inline void *
Region::end() const
{
    return __void(_addr + _size);
}

inline void
Region::relate(Context *ctx)
{
    lockWrite();
    _relatives.push_back(ctx);
    unlock();
}

inline void
Region::unrelate(Context *ctx)
{
    lockWrite();
    _relatives.remove(ctx);
    unlock();
}

inline void
Region::transfer()
{
    ASSERT(_relatives.empty() == false);
    lockWrite();
    _context = _relatives.front();
    _relatives.pop_front();
    unlock();
}

inline std::list<Context *> &
Region::relatives()
{
    return _relatives;
}

#endif
