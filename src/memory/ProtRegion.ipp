#ifndef __MEMORY_PROTREGION_IPP_
#define __MEMORY_PROTREGION_IPP_

namespace gmac { namespace memory {

inline void
ProtRegion::invalidate(void)
{
   assertion(tryWrite() == false);
   _present = _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_NONE);
   assertion(ret == 0);
}

inline void
ProtRegion::readOnly(void)
{
   assertion(tryWrite() == false);
   _present = true;
   _dirty = false;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ);
   assertion(ret == 0);
}

inline void
ProtRegion::readWrite(void)
{
   assertion(tryWrite() == false);
   _present = _dirty = true;
   int ret = Memory::protect(__void(_addr), _size, PROT_READ | PROT_WRITE);
   assertion(ret == 0);
}

inline bool
ProtRegion::dirty()
{
   assertion(tryWrite() == false);
   return _dirty;
}

inline bool
ProtRegion::present()
{
   assertion(tryWrite() == false);
   return _present;
}

}}

#endif
