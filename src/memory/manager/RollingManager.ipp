#ifndef __MEMOMRY_ROLLINGMANAGER_IPP_
#define __MEMOMRY_ROLLINGMANAGER_IPP_

inline bool
RollingBuffer::overflows() const
{
    return _buffer.size() >= _max;
}

inline size_t
RollingBuffer::inc(size_t n)
{
    _max += n;
}

inline size_t
RollingBuffer::dec(size_t n)
{
    _max -= n;
}

inline bool
RollingBuffer::empty() const
{
    return _buffer.empty();
}

inline void
RollingBuffer::push(RollingBlock *region)
{
    _lock.lockWrite();
    _buffer.push_back(region);
    _lock.unlock();
}

inline RollingBlock *
RollingBuffer::pop()
{
    _lock.lockWrite();
    assert(_buffer.empty() == false);
    RollingBlock *ret = _buffer.front();
    _buffer.pop_front();
    _lock.unlock();
    return ret;
}

inline RollingBlock *
RollingBuffer::front()
{
    _lock.lockRead();
    RollingBlock *ret = _buffer.front();
    _lock.unlock();
    return ret;
}

inline void
RollingBuffer::remove(RollingBlock *region)
{
    _lock.lockWrite();
    _buffer.remove(region);
    _lock.unlock();
}

inline size_t
RollingBuffer::size() const
{
    return _buffer.size();
}

inline void
RollingManager::invalidate(RollingBlock *region)
{
    regionRolling[Context::current()]->remove(region);
}

inline void
RollingManager::flush(RollingBlock *region)
{
    regionRolling[Context::current()]->remove(region);
    assert(region->copyToDevice() == gmacSuccess);
}

#endif
