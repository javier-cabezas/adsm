#ifndef __VM_IPP_
#define __VM_IPP_

template<typename T>
T *
Table<T>::entry(size_t n) const
{
    assert(n < nEntries);
    return (T *)((addr_t)table[n] & Mask);
}

template<typename T>
Table<T>::Table(size_t nEntries) :
    nEntries(nEntries)
#ifdef USE_VM
    , __shared(true)
#endif
{
    TRACE("Creating Table with %d entries (%p)", nEntries, this);

    assert(posix_memalign((void **)&table, 0x1000,
                nEntries * sizeof(T *)) == 0);
    memset(table, 0, nEntries * sizeof(T *));
#ifdef USE_VM
#ifdef USE_VM_DEVICE
    assert(posix_memalign((void **)&__shadow, 0x1000,
                nEntries * sizeof(T *)) == 0);
    __device = Dumper::alloc(nEntries * sizeof(T *));
#else
    __device = Dumper::hostAlloc((void **)&__shadow, nEntries * sizeof(T *));
#endif
    if(__shadow != NULL) memset(__shadow, 0, nEntries * sizeof(T *));
#endif
}

template<typename T>
Table<T>::~Table()
{
    TRACE("Cleaning Table with %d entries (%p)", nEntries, this);
    free(table);
#ifdef USE_VM
    enterFunction(vmFree);
#ifdef USE_VM_DEVICE
    free(__shadow);
    if(__device != NULL) Dumper::free(__device);
#else
    if(__shadow != NULL) Dumper::hostFree(__shadow);
#endif
    exitFunction();
#endif
}

template<typename T>
inline bool
Table<T>::present(size_t n) const
{
    assert(n < nEntries);
    return (addr_t)table[n] & Present;
}

template<typename T>
inline bool
Table<T>::dirty(size_t n) const
{
    assert(n < nEntries);
    return (addr_t)table[n] & Dirty;
}

template<typename T>
inline void
Table<T>::clean(size_t n){
    assert(n < nEntries);
    table[n] = (addr_t *)((addr_t)table[n] & ~Dirty);
}

#ifdef USE_VM
template<typename T>
void *
Table<T>::device()
{
    return __device;
}
#endif

template<typename T>
void
Table<T>::create(size_t n, size_t size)
{
    enterFunction(vmAlloc);
    assert(n < nEntries);
    table[n] = (T *)((addr_t)new T(size) | Present);
#ifdef USE_VM
    __shared = false;
    __shadow[n] = (T *)((addr_t)entry(n)->device() | Present);
#endif
    exitFunction();
}

template<typename T>
void
Table<T>::insert(size_t n, void *addr)
{
    assert(n < nEntries);
    table[n] = (T *)((addr_t)addr | Present);
#ifdef USE_VM
    __shadow[n] = (T *)((addr_t)addr | Present);
#endif
}

template<typename T>
void
Table<T>::remove(size_t n)
{
    assert(n < nEntries);
    table[n] = (T *)0;
#ifdef USE_VM
    __shadow[n] = (T *)0;
#endif
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

template<typename T>
inline void
Table<T>::realloc()
{
#ifdef USE_VM
#ifdef USE_VM_DEVICE
    if(__device != NULL) Dumper::free(__device);
    __device = Dumper::alloc(nEntries * sizeof(T *));
#else
    if(__device != NULL) Dumper::hostFree(__shadow);
    __device = Dumper::hostAlloc((void **)&__shadow, nEntries * sizeof(T *));
    memset(__shadow, 0, nEntries * sizeof(T *));
#endif
    assert(__device != NULL);
#endif
}

template<typename T>
inline void
Table<T>::flush() const
{
#ifdef USE_VM
		assert(__device != NULL);
		Dumper::flush(__device, __shadow, nEntries * sizeof(T *));
#endif
}

template<typename T>
inline void
Table<T>::sync() const
{
#ifdef USE_VM
    assert(__device != NULL);
    if(__shared == false) return;
    Dumper::sync(table, __device, nEntries * sizeof(T *));
#endif
}

#endif
