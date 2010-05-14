#ifndef __MEMORY_PROTREGION_IPP_
#define __MEMORY_PROTREGION_IPP_

namespace gmac { namespace memory {

inline void
ProtRegion::invalidate(void)
{
   logger.assertion(tryWrite() == false);
   _present = _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_NONE);
   logger.assertion(ret == 0);
}

inline void
ProtRegion::readOnly(void)
{
   logger.assertion(tryWrite() == false);
   _present = true;
   _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ);
   logger.assertion(ret == 0);
}

inline void
ProtRegion::readWrite(void)
{
   logger.assertion(tryWrite() == false);
   _present = _dirty = true;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ | PROT_WRITE);
   logger.assertion(ret == 0);
}

inline bool
ProtRegion::dirty()
{
   logger.assertion(tryWrite() == false);
   return _dirty;
}

inline bool
ProtRegion::present()
{
   logger.assertion(tryWrite() == false);
   return _present;
}

}}

#endif
