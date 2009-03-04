#include "StatManager.h"


namespace gmac {

std::ostream &operator<<(std::ostream &os, const Event::Type &t)
{
	switch(t) {
		case Event::Alloc:
			os << "Alloc"; break;
		case Event::Free:
			os << "Free"; break;
		case Event::Execute:
			os << "Execute"; break;
		case Event::Sync:
			os << "Sync"; break;
	}
	return os;
}

std::ostream &operator<<(std::ostream &os, const Event &e)
{
	os << e.time << ":" << e.type << ":";
	if(e.addr) os << e.addr;
	os << ":";
	if(e.size) os << e.size;
	os << ":" << std::endl;
	return os;
}

StatManager::~StatManager()
{
	std::list<Event>::const_iterator e;
	events.sort();
	for(e = events.begin(); e != events.end(); e++)
		std::cout << *e;
}

};
