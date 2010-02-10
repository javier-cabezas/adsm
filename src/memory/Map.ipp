#ifndef __MEMORY_MAP_IPP_
#define __MEMORY_MAP_IPP_

inline void
Map::realloc()
{
    __pageTable.realloc();
}

inline void
Map::lock()
{
    local.read();
}

inline void
Map::unlock()
{
    local.unlock();
}

inline Map::iterator
Map::begin()
{
    return __map.begin();
}

inline Map::iterator
Map::end()
{
    return __map.end();
}

inline void
Map::insert(Region *i)
{
    local.write();
    __map.insert(__Map::value_type(i->end(), i));
    local.unlock();

    global.write();
    __global->insert(__Map::value_type(i->end(), i));
    global.unlock();
}

inline PageTable &
Map::pageTable()
{
    return __pageTable;
}

inline const PageTable &
Map::pageTable() const
{
    return __pageTable;
}

template<typename T>
inline T *
Map::find(const void *addr)
{
    Region *ret = NULL;
    local.read();
    ret = localFind(addr);
    if(ret == NULL) {
        global.read();
        ret = globalFind(addr);
        global.unlock();
    }
    local.unlock();
    return dynamic_cast<T *>(ret);
}

#endif
