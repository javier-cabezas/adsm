#ifndef __MEMOMRY_ROLLINGMANAGER_IPP_
#define __MEMOMRY_ROLLINGMANAGER_IPP_

inline bool
RollingBuffer::overflows() const
{
    return buffer.size() >= _max;
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
    return buffer.empty();
}

inline void
RollingBuffer::push(ProtSubRegion *region)
{
    lock.write();
    buffer.push_back(region);
    lock.unlock();
}

inline ProtSubRegion *
RollingBuffer::pop()
{
    lock.write();
    assert(buffer.empty() == false);
    ProtSubRegion *ret = buffer.front();
    buffer.pop_front();
    lock.unlock();
    return ret;
}

inline ProtSubRegion *
RollingBuffer::front()
{
    lock.read();
    ProtSubRegion *ret = buffer.front();
    lock.unlock();
    return ret;
}

inline void
RollingBuffer::remove(ProtSubRegion *region)
{
    lock.write();
    buffer.remove(region);
    lock.unlock();
}

inline void
RollingManager::invalidate(ProtSubRegion *region)
{
    regionRolling[Context::current()]->remove(region);
}

inline void
RollingManager::flush(ProtSubRegion *region)
{
	regionRolling[Context::current()]->remove(region);
	assert(region->copyToDevice() == gmacSuccess);
}

#endif
