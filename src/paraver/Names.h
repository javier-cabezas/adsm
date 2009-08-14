/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef __PARAVER_STATENAME_H_
#define __PARAVER_STATENAME_H_

#ifndef GENERATE_PARAVER_STRING
	#define STATE(name, val) extern StateName *name
	#define EVENT(name, val) extern EventName *name
#else
	#define STATE(name, val) \
		StateName *name; \
		static StateFactory __attribute__((init_priority(101))) name##Factory(name, #name, val)
 	#define EVENT(name, val) \
		EventName *name; \
		static EventFactory __attribute__((init_priority(101))) name##Factory(name, #name, val)
#endif

#include <string>
#include <vector>
#include <map>

namespace paraver {
class Name {
private:
	std::string name;
	int32_t value;
public:
	Name(const char *name, int32_t value) :
		name(std::string(name)),
		value(value)
	{};
	inline std::string getName() const { return name; }
	inline int32_t getValue() const { return value; }
};

class StateName : public Name {
public:
	typedef std::vector<const StateName *> List;
protected:
	static List *states;
public:
	StateName(const char *name, int32_t value) : Name(name, value) {
		if(states == NULL) states = new List();
		states->push_back(this);
	}
	static const std::vector<const StateName *> &get() { return *states; }
	static void destroy() {
		List::const_iterator i;
		for(i = states->begin(); i != states->end(); i++)
			delete *i;
		delete states;
	}
};

class StateFactory {
public:
	StateFactory(StateName *&s, const char *name, int32_t value) {
		s = new StateName(name, value);
	}
};

class EventName : public Name {
public:
	typedef std::vector<const EventName *> List;
	typedef std::map<uint32_t, std::string> TypeTable;
protected:
	static List *events;
	TypeTable types;
public:
	EventName(const char *name, int32_t value) : Name(name, value) {
		if(events == NULL) events = new List();
		events->push_back(this);
	}
	static const List &get() { return *events; }
	static void destroy() { 
		List::const_iterator i;
		for(i = events->begin(); i != events->end(); i++)
			delete *i;
		delete events;
	}
	void registerType(uint32_t value, std::string type) {
		types.insert(TypeTable::value_type(value, type));
	}
	const TypeTable &getTypes() const { return types; }
};

class EventFactory {
public:
	EventFactory(EventName *&e, const char *name, int32_t value) {
		e = new EventName(name, value);
	}
};

};

#endif
