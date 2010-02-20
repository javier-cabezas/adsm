#ifndef __MEMORY_PROTREGION_IPP_
#define __MEMORY_PROTREGION_IPP_

namespace gmac { namespace memory {

inline void
ProtRegion::invalidate(void)
{
   ASSERT(tryWrite() == false);
   _present = _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_NONE);
   ASSERT(ret == 0);
}

inline void
ProtRegion::readOnly(void)
{
   ASSERT(tryWrite() == false);
   _present = true;
   _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ);
   ASSERT(ret == 0);
}

inline void
ProtRegion::readWrite(void)
{
   ASSERT(tryWrite() == false);
   _present = _dirty = true;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ | PROT_WRITE);
   ASSERT(ret == 0);
}

inline bool
ProtRegion::dirty()
{
   ASSERT(tryWrite() == false);
   return _dirty;
}

inline bool
ProtRegion::present()
{
   ASSERT(tryWrite() == false);
   return _present;
}

}}

#endif
