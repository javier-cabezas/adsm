#ifndef GMAC_CORE_HPE_PROCESS_IMPL_H_
#define GMAC_CORE_HPE_PROCESS_IMPL_H_

namespace __impl { namespace core { namespace hpe {

#if 0
inline size_t
process::nAccelerators() const
{
    return accs_.size();
}

inline Accelerator &
process::getAccelerator(unsigned i)
{
    ASSERTION(i < accs_.size(), "Incorrect accelerator ID");

    return *accs_[i];
}
#endif

inline memory::Protocol *
process::getProtocol()
{
    return &protocol_;
}

inline memory::map_object &
process::shared()
{
    return shared_;
}

inline const memory::map_object &
process::shared() const
{
    return shared_;
}

inline memory::map_object &
process::global()
{
    return global_;
}

inline const memory::map_object &
process::global() const
{
    return global_;
}

inline memory::map_object &
process::orphans()
{
    return orphans_;
}

inline const memory::map_object &
process::orphans() const
{
    return orphans_;
}

inline void
process::makeOrphan(memory::object &obj)
{
    TRACE(LOCAL, "Making orphan object: %p", obj.addr());
    // Insert into the orphan list
    orphans_.addObject(obj);
    // Remove from the list of regular shared objects
    shared_.removeObject(obj);
}

inline resource_manager &
process::get_resource_manager()
{
    return resourceManager_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
