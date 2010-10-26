#include <string>

#include "Lock.h"

namespace gmac { namespace util {

#ifdef PARAVER
const char *ParaverLock::eventName = "Lock";
const char *ParaverLock::exclusiveName = "Exclusive";
ParaverLock::LockMap *ParaverLock::map = NULL;
paraver::EventName *ParaverLock::event = NULL;
paraver::StateName *ParaverLock::exclusive = NULL;
#endif

ParaverLock::ParaverLock(const char *name) 
{
	UNREFERENCED_PARAMETER(name);
#ifdef PARAVER
    if(event == NULL)
        event = paraver::Factory<paraver::EventName>::create(eventName);

    if(map == NULL) map = new LockMap();
    LockMap::const_iterator i = map->find(std::string(name));
    if(i == map->end()) {
        id = map->size() + 1;
        event->registerType(id, std::string(name));
        map->insert(LockMap::value_type(std::string(name), id));
    }
    else id = i->second;

    if(exclusive == NULL)
        exclusive = paraver::Factory<paraver::StateName>::create(exclusiveName);
#endif
}
}}
