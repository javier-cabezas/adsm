#ifndef GMAC_CORE_PROCESS_IMPL_H_
#define GMAC_CORE_PROCESS_IMPL_H_

#include "core/Map.h"

namespace __impl { namespace core {

inline size_t
Process::nAccelerators() const
{
    return accs_.size();
}
    
inline Accelerator *
Process::getAccelerator(unsigned i)
{
    if (i >= accs_.size()) return NULL;

    return accs_[i];
}

inline memory::Protocol &
Process::protocol()
{
    return protocol_;
}

inline memory::ObjectMap &
Process::shared()
{
    return shared_;
}

inline const memory::ObjectMap &
Process::shared() const
{
    return shared_;
}

inline memory::ObjectMap &
Process::global()
{
    return global_;
}

inline const memory::ObjectMap &
Process::global() const
{
    return global_;
}

inline memory::ObjectMap &
Process::orphans()
{
    return orphans_;
}

inline const memory::ObjectMap &
Process::orphans() const
{
    return orphans_;
}

inline void
Process::insertOrphan(memory::Object &obj)
{
    Map::insertOrphan(obj);
}

}}

#endif
