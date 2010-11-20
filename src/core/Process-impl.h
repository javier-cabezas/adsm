#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

namespace __impl { namespace core { 

inline size_t
Process::totalMemory()
{
    return TotalMemory_;
}

inline size_t
Process::nAccelerators() const
{
    return accs_.size();
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

}}

#endif
