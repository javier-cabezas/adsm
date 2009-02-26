#ifndef __STATMANAGER_H_
#define __STATMANAGER_H_

#include "MemManager.h"

#include <time.h>
#include <list>
#include <iostream>

namespace icuda {

class Event {
public:
	typedef enum { Alloc, Free, Execute, Sync } Type;
protected:
	Type type;
	void *addr;
	size_t size;
	clock_t time;
public:
	Event(Type type, void *addr = NULL, size_t size = 0) :
		type(type),
		addr(addr),
		size(size),
		time(clock()) {}

	bool operator<(const Event &e) const {
		return time < e.time;
	}

	friend std::ostream &operator<<(std::ostream &os, const Event &e);
	friend std::ostream &operator<<(std::ostream &os, const Type &e);

};

class StatManager : public MemManager {
protected:
	std::list<Event> events;
public:
	virtual ~StatManager();
	virtual bool alloc(void *addr, size_t count) {
		events.push_back(Event(Event::Alloc, addr, count));
	}
	virtual void release(void *addr) {
		events.push_back(Event(Event::Free, addr));
	}
	virtual void execute(void) {
		events.push_back(Event(Event::Execute));
	}
	virtual void sync(void) {
		events.push_back(Event(Event::Sync));
	}
};

};

#endif
