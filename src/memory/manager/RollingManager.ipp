#ifndef __MEMOMRY_ROLLINGMANAGER_IPP_
#define __MEMOMRY_ROLLINGMANAGER_IPP_

inline bool
RollingBuffer::overflows() const
{
   bool ret = _buffer.size() >= _max;
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
   _buffer.push_back(region);
}

inline RollingBlock *
RollingBuffer::pop()
{
   assert(_buffer.empty() == false);
   RollingBlock *ret = _buffer.front();
   _buffer.pop_front();
   return ret;
}

inline RollingBlock *
RollingBuffer::front()
{
   RollingBlock *ret = _buffer.front();
   return ret;
}

inline void
RollingBuffer::remove(RollingBlock *region)
{
   _buffer.remove(region);
}

inline size_t
RollingBuffer::size() const
{
   return _buffer.size();
}

inline RollingBuffer *
RollingMap::createBuffer(Context *ctx)
{
   RollingBuffer *ret = new RollingBuffer();
   lockWrite();
   this->insert(RollingMap::value_type(ctx, ret)); 
   unlock();
   return ret;
}

inline RollingBuffer *
RollingMap::currentBuffer()
{
   RollingBuffer *ret = NULL;
   lockRead();
   RollingMap::iterator i = this->find(Context::current());
   if(i != this->end()) ret = i->second;
   unlock();
   if(ret == NULL) ret = createBuffer(Context::current());
   return ret;
}

inline RollingBuffer *
RollingMap::contextBuffer(Context *ctx)
{
   RollingBuffer *buffer = NULL;
   lockWrite();
   RollingMap::iterator i = this->find(ctx);
   if(i != this->end()) buffer = i->second;
   unlock();
   if(buffer== NULL) buffer = createBuffer(ctx);
   return buffer;
}

inline void
RollingMap::remove(RollingBlock *block)
{
   lockWrite();
   RollingMap::iterator i;
   for(i = this->begin(); i != this->end(); i++) {
      i->second->lockWrite();
      i->second->remove(block);
      i->second->unlock();
   }
   unlock();
}

inline void
RollingManager::invalidate(RollingBlock *region)
{
   rollingMap.remove(region);

   region->lockWrite();
   region->invalidate();
   region->unlock();
}

inline void
RollingManager::flush(RollingBlock *region)
{
   rollingMap.remove(region);
   region->lockWrite();
   if(region->dirty()) {
      assert(region->copyToDevice() == gmacSuccess);
      region->readOnly();
   }
   region->unlock();
}

#endif
