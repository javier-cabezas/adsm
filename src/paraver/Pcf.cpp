#include "Pcf.h"
#include <paraver/Types.h>
#include <values.h>

#include <vector>

namespace paraver {

std::ostream &pcf(std::ostream &os)
{
	StateName::List::const_iterator s;
	int long long max = MININT, min = MAXINT;
	for(s = StateName::get().begin(); s != StateName::get().end(); s++) {
		max = (max > (*s)->getValue()) ? max : (*s)->getValue();
		min = (min < (*s)->getValue()) ? min : (*s)->getValue();
	}
	if(max <= min) max = min + 1;

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
	os << "0 {0, 0, 0}" << std:: endl;
	for(int i = 1; i < 20; i++) {
		int r = 54 - (3 * i);
		int g = 40 * i - (2 * i * i) - 38;
		int b = 255 * i / 19;
		os << i << " {" << r << ", " << g << ", " << b << "}" << std:: endl;
	}
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
