#include "Element.h"

namespace paraver {

void Task::write(std::ofstream &of) const
{
	uint32_t id = node->getId();
	// We need to store the node where the task was binded
	of.write((char *)&id, sizeof(id));
	Element<Application, Thread>::write(of);
}

std::ostream &operator<<(std::ostream &os,
		const Application &app) 
{
	HASH_MAP<int32_t, Task *>::const_iterator i;
	os << "(";
	for(i = app.sons.begin(); i != app.sons.end(); i++) {
		if(i != app.sons.begin()) os << ",";
		os << i->second->getId() << ":" << i->second->size();
	}
	os << ")";
	return os;
}
};
