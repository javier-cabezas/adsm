#ifndef __MEMORY_PAGETABLE_IPP_
#define __MEMORY_PAGETABLE_IPP_

inline int
PageTable::entry(const void *addr, unsigned long shift, size_t size) const
{
    unsigned long n = (unsigned long)(addr);
    return (n >> shift) & (size - 1);
}

inline int
PageTable::offset(const void *addr) const
{
    unsigned long n = (unsigned long)(addr);
    return n & (paramPageSize - 1);
}

inline void
PageTable::realloc()
{
    rootTable.realloc();
}

inline const void *
PageTable::translate(const void *host)
{
    return translate((void *)host);
}

inline size_t
PageTable::getPageSize() const
{
    return paramPageSize;
}

inline size_t
PageTable::getTableShift() const
{
    return tableShift;
}

inline size_t
PageTable::getTableSize() const
{
    return (1 << dirShift) / paramPageSize;
}

inline void
PageTable::invalidate()
{
    _valid = false;
}

#endif
