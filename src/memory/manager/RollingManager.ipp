#ifndef __MEMOMRY_ROLLINGMANAGER_IPP_
#define __MEMOMRY_ROLLINGMANAGER_IPP_

namespace gmac { namespace memory { namespace manager {

inline int
RollingManager::defaultProt()
{
    return PROT_READ;
}

inline bool
RollingBuffer::overflows() 
{
   lockRead();
   bool ret = _buffer.size() >= _max;
   unlock();
   return ret;
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
RollingBuffer::empty() 
{
   lockRead();
   bool ret = _buffer.empty();
   unlock();
   return ret;
}

inline void
RollingBuffer::push(RollingBlock *region)
{
   lockWrite();
   _buffer.push_back(region);
   unlock();
}

inline RollingBlock *
RollingBuffer::pop()
{
   lockWrite();
   ASSERT(_buffer.empty() == false);
   RollingBlock *ret = _buffer.front();
   _buffer.pop_front();
   unlock();
   return ret;
}

inline RollingBlock *
RollingBuffer::front()
{
   lockRead();
   RollingBlock *ret = _buffer.front();
   unlock();
   return ret;
}

inline void
RollingBuffer::remove(RollingBlock *region)
{
   lockWrite();
   _buffer.remove(region);
   unlock();
}

inline size_t
RollingBuffer::size()
{
   lockRead();
   size_t ret = _buffer.size();
   unlock();
   return ret;
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
      i->second->remove(block);
   }
   unlock();
}

#if 0
inline void
RollingManager::invalidate(RollingBlock *block)
{
   rollingMap.remove(block);
   block->invalidate();
}
#endif

inline void
RollingManager::flush(RollingBlock *block)
{
   rollingMap.remove(block);
   ASSERT(block->dirty() == true);
   gmacError_t ret = block->copyToDevice();
   ASSERT(ret == gmacSuccess);
   block->readOnly();
}

inline void
RollingManager::forceFlush(RollingBlock *block)
{
   gmacError_t ret = block->copyToDevice();
   ASSERT(ret == gmacSuccess);
}

}}}

#endif
