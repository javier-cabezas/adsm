#ifndef GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_LAZY_IMPL_H_


namespace __impl { namespace memory { namespace protocol { 

inline BlockList::BlockList() :
    Lock("BlockList")
{}

inline BlockList::~BlockList()
{}

inline bool BlockList::empty() const
{
    lock();
    bool ret = Parent::empty();
    unlock();
    return ret;
}

inline size_t BlockList::size() const
{
    lock();
    size_t ret = Parent::size();
    unlock();
    return ret;
}

inline void BlockList::push(Block &block)
{
    lock();
    Parent::push_back(&block);
    unlock();
}

inline Block *BlockList::pop()
{
    Block *ret = (Block *)NULL;
    lock();
    if(Parent::empty() == false) {
        ret = Parent::front();
        Parent::pop_front();
    }
    unlock();
    return ret;
}

inline BlockListMap::BlockListMap() :
    gmac::util::RWLock("BlockListMap")
{}

inline BlockListMap::~BlockListMap()
{
    lockWrite();
    iterator i;
    for(i = Parent::begin(); i != Parent::end(); i++)
        delete i->second;
    Parent::clear();
    unlock();
}

inline const BlockList *BlockListMap::get(core::Mode *mode) const
{
    const BlockList *ret = NULL;
    lockRead();
    const_iterator i = Parent::find(mode);
    if(i != Parent::end()) ret = i->second;
    unlock();
    return ret;
}

inline BlockList *BlockListMap::get(core::Mode *mode)
{
    lockWrite();
    iterator i = Parent::find(mode);
    if(i == Parent::end()) {
        std::pair<iterator, bool> pair = 
            Parent::insert(Parent::value_type(mode, new BlockList()));
        i = pair.first;
    }
    BlockList *ret = i->second;
    unlock();
    return ret;
}

inline bool BlockListMap::empty(core::Mode &mode) const
{
    const BlockList *list = get(&mode);
    if(list == NULL) return true;
    return list->empty();
}

inline size_t BlockListMap::size(core::Mode &mode) const
{
    const BlockList *list = get(&mode);
    if(list == NULL) return 0;
    return list->size();
}

inline void BlockListMap::push(core::Mode &mode, Block &block)
{
    BlockList *list = get(&mode);
    return list->push(block);
}

inline Block *BlockListMap::pop(core::Mode &mode)
{
    BlockList *list = get(&mode);
    if(list == NULL) return NULL;
    return list->pop();
}

}}}

#endif
