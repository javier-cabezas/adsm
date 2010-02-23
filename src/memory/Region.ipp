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

inline bool
Region::shared()
{
    return _shared;
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
    _relatives.push_back(ctx);
}

inline void
Region::unrelate(Context *ctx)
{
    _relatives.remove(ctx);
}

inline void
Region::transfer()
{
    ASSERT(_relatives.empty() == false);
    _context = _relatives.front();
    _relatives.pop_front();
}

inline std::list<Context *> &
Region::relatives()
{
    return _relatives;
}

#endif
