#include "hpe/core/thread.h"

namespace __impl { namespace core { namespace hpe {

thread::~thread()
{
    for (map_config::iterator it = mapConfigs_.begin(); it != mapConfigs_.end(); ++it) {
        delete it->second.first;
        delete it->second.second;
    }

#if 0
    for (map_context::iterator it = mapContexts_.begin(); it != mapContexts_.end(); it++) {
        delete it->second;
    }
#endif

    for (map_vdevice::iterator it = mapVDevices_.begin(); it != mapVDevices_.end(); ++it) {
        process_.get_resource_manager().destroy_virtual_device(*it->second);
    }
}

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
