#ifndef __MEMORY_PROTREGION_IPP_
#define __MEMORY_PROTREGION_IPP_

inline void
ProtRegion::invalidate(void)
{
   assert(tryWrite() == false);
   _present = _dirty = false;
   assert(Memory::protect(__void(_addr), _size, PROT_NONE) == 0);
}

inline void
ProtRegion::readOnly(void)
{
   assert(tryWrite() == false);
   _present = true;
   _dirty = false;
   assert(Memory::protect(__void(_addr), _size, PROT_READ) == 0);
}

inline void
ProtRegion::readWrite(void)
{
   assert(tryWrite() == false);
   _present = _dirty = true;
   assert(Memory::protect(__void(_addr), _size, PROT_READ | PROT_WRITE) == 0);
}

inline bool
ProtRegion::dirty()
{
   assert(tryRead() == false);
   return _dirty;
}

inline bool
ProtRegion::present()
{
   assert(tryRead() == false);
   return _present;
}

#endif
