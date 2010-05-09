#ifndef __MEMORY_TABLE_IPP_
#define __MEMORY_TABLE_IPP_

namespace gmac { namespace memory  { namespace vm {

template<typename T>
T *
Table<T>::entry(size_t n) const
{
    ASSERT(n < nEntries);
    return (T *)((addr_t)table[n] & Mask);
}

template<typename T>
Table<T>::Table(size_t nEntries) :
    nEntries(nEntries)
{
    TRACE("Creating Table with %zd entries (%p)", nEntries, this);

    /* int ret = 0;
     * ASSERT(posix_memalign((void **)&table, 0x1000,
                nEntries * sizeof(T *)) == 0); */
    table = (T **)valloc(nEntries * sizeof(T *));
    ASSERT(table != NULL);
    memset(table, 0, nEntries * sizeof(T *));
    TRACE("Table memory @ %p", table);
}

template<typename T>
Table<T>::~Table()
{
    TRACE("Cleaning Table with %zd entries (%p) @ %p", nEntries, this, table);
    ::free(table);
}

template<typename T>
inline bool
Table<T>::present(size_t n) const
{
    ASSERT(n < nEntries);
    return (addr_t)table[n] & Present;
}

template<typename T>
inline bool
Table<T>::dirty(size_t n) const
{
    ASSERT(n < nEntries);
    return (addr_t)table[n] & Dirty;
}

template<typename T>
inline void
Table<T>::clean(size_t n){
    ASSERT(n < nEntries);
    table[n] = (addr_t *)((addr_t)table[n] & ~Dirty);
}


template<typename T>
void
Table<T>::create(size_t n, size_t size)
{
    enterFunction(FuncVmAlloc);
    ASSERT(n < nEntries);
    table[n] = (T *)((addr_t)new T(size) | Present);
    exitFunction();
}

template<typename T>
void
Table<T>::insert(size_t n, void *addr)
{
    ASSERT(n < nEntries);
    table[n] = (T *)((addr_t)addr | Present);
}

template<typename T>
void
Table<T>::remove(size_t n)
{
    ASSERT(n < nEntries);
    table[n] = (T *)0;
}

template<typename T>
inline T &
Table<T>::get(size_t n) const
{
    return *entry(n);
}

template<typename T>
inline T *
Table<T>::value(size_t n) const
{
    return entry(n);
}

template<typename T>
inline size_t
Table<T>::size() const
{
    return nEntries;
}


}}}

#endif
