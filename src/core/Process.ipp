#ifndef GMAC_CORE_MODE_IPP_
#define GMAC_CORE_MODE_IPP_

namespace gmac {

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
Process::replicated()
{
    return replicated_;
}

inline const memory::ObjectMap &
Process::replicated() const
{
    return replicated_;
}

inline memory::ObjectMap &
Process::centralized()
{
    return centralized_;
}

inline const memory::ObjectMap &
Process::centralized() const
{
    return centralized_;
}

}

#endif
