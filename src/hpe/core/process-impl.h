#ifndef GMAC_CORE_HPE_PROCESS_IMPL_H_
#define GMAC_CORE_HPE_PROCESS_IMPL_H_

namespace __impl { namespace core { namespace hpe {

inline resource_manager &
process::get_resource_manager()
{
    return resourceManager_;
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
