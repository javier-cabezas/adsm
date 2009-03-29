#include "Pcf.h"
#include <values.h>

#include <vector>

namespace paraver {

std::ostream &pcf(std::ostream &os)
{
	StateName::List::const_iterator s;
	int max = MININT, min = MAXINT;
	for(s = StateName::get().begin(); s != StateName::get().end(); s++) {
		max = (max > (*s)->getValue()) ? max : (*s)->getValue();
		min = (min < (*s)->getValue()) ? min : (*s)->getValue();
	}

	os << "DEFAULT_OPTIONS" << std::endl;
	os << "YMAX_SCALE " << max << std::endl;
	os << "YMIN_SCALE " << min << std::endl;
	os << std::endl;

	os << "DEFAULT_SEMANTIC" << std::endl;
	os << std::endl;

	os << "THREAD_FUNC State As Is" << std::endl;
	os << std::endl;

	if(StateName::get().empty() == false) {
		os << "STATES" << std::endl;
		for(s = StateName::get().begin(); s != StateName::get().end(); s++)
			os << (*s)->getValue() << " " << (*s)->getName() << std::endl;
		os << std::endl;
	}

	os << std::endl;
	os << "GRADIENT_COLOR" << std::endl;
	for(int i = 0; i < 20; i++)
		os << i << " {" << 63 + (i * 192 / 19) << ", 0, 0}" << std:: endl;
	os << std::endl;

	if(EventName::get().empty() == false) {
		os << std::endl;
		EventName::List::const_iterator e;
		int i = 0;
		for(e = EventName::get().begin(); e != EventName::get().end(); e++) {
			os << "EVENT_TYPE" << std::endl;
			os << ++i << " " << (*e)->getValue() << " " << (*e)->getName() << std::endl;
			const EventName::TypeTable &types = (*e)->getTypes();
			if(types.empty()) continue;
			EventName::TypeTable::const_iterator t;
			os << std::endl << "VALUES" << std::endl;
			for(t = types.begin(); t != types.end(); t++) {
				os << t->first << " " << t->second << std::endl;
			}
			os << std::endl;
		}
		os << std::endl;
	}

	return os;
}

};
