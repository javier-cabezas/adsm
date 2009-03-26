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

#ifndef __PARAVER_ELEMENT_H_
#define __PARAVER_ELEMENT_H_

#include <config/config.h>
#include <paraver/Time.h>
#include <paraver/Record.h>

#include <assert.h>

#include <typeinfo>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

namespace paraver {
class Abstract {
protected:
	int32_t id;
	std::string name;
public:
	Abstract(int32_t id, std::string name) : id(id), name(name) {};
	inline int32_t getId() const { return id; }
	std::string getName() const {
		std::ostringstream stream;
		stream << name << id;
		return stream.str();
	}

	virtual inline void end(std::ofstream &of, Time_t t) = 0;
};

template<typename P, typename S>
class Element : public Abstract {
protected:
	P *parent;
	HASH_MAP<int32_t, S *> sons;

	inline P *getParent() const { return parent; }
	inline void addSon(int32_t id, S *son) { sons[id] = son; }
	inline S *getSon(int32_t id) const {
		typename HASH_MAP<int32_t, S *>::const_iterator i;
		i = sons.find(id);
		assert(i != sons.end());
		return i->second;
	}
public:
	Element(P *parent, int32_t id, std::string name) :
		Abstract(id, name), parent(parent) { };
	virtual ~Element() {
		if(typeid(S) == typeid(void)) return;
		typename HASH_MAP<int32_t, S *>::const_iterator i;
		for(i = sons.begin(); i != sons.end(); i++)
			if(i->second) delete i->second;
	}
	inline size_t size() const { return sons.size(); }
	inline virtual void end(std::ofstream &os, Time_t t) {
		typename HASH_MAP<int32_t, S *>::const_iterator i;
		for(i = sons.begin(); i != sons.end(); i++)
			i->second->end(os, t);
	}
	inline virtual void write(std::ofstream &os) const {
		os.write((char *)&id, sizeof(id));
		uint32_t s = size();
		os.write((char *)&s, sizeof(s));
		typename HASH_MAP<int32_t, S *>::const_iterator i;
		for(i = sons.begin(); i != sons.end(); i++)
			i->second->write(os);
	}
};

template<typename P>
class Element<P, void> : public Abstract {
protected:
	P *parent;
	inline P *getParent() const { return parent; }
public:
	Element(P *parent, int32_t id, std::string name) :
		Abstract(id, name), parent(parent) { };
	virtual ~Element() { }
	inline size_t size() const { return 0; }
	inline void write(std::ofstream &of) const {
		of.write((char *)&id, sizeof(id));
		uint32_t s = size();
		of.write((char *)&s, sizeof(s));
	}
};

class Application;
class Task;
class Thread : public Element<Task, void> {
protected:
	std::vector<State> states;
public:
	Thread(Task *task, int32_t id) :
		Element<Task, void>(task, id, "Thread")
	{
		states.push_back(State(this));
		states.back().start(State::None, 0);
	}

	void start(std::ofstream &of, unsigned s, Time_t t);
	void end(std::ofstream &of, Time_t t);

	inline int32_t getTask() const;
	inline int32_t getApp() const;

	std::string print() const;
};


class Task : public Element<Application, Thread> {
protected:
	int32_t threads;
public:
	Task(Application *app, int32_t id) :
		Element<Application, Thread>(app, id, "Task"), threads(1)
	{};

	inline Thread *__addThread(int32_t id) {
		Thread *thread = new Thread(this, threads++);
		addSon(id, thread);
		return thread;
	}
	inline Thread *getThread(int32_t id) const { return getSon(id); }
	inline int32_t getApp() const;

};


class Application : public Element<void, Task> {
protected:
	int32_t tasks;
public:
	Application(uint32_t id, std::string name)
		: Element<void, Task>(NULL, id, name), tasks(1)
	{};

	inline Task *getTask(int32_t id) const { return getSon(id); }
	inline Task *addTask(int32_t id) {
		Task *task = new Task(this, tasks++);
		addSon(id, task);
		return task;
	}

	friend std::ostream &operator<<(std::ostream &os, const Application &app);
};

inline int32_t Thread::getTask() const { return parent->getId(); }
inline int32_t Thread::getApp() const { return parent->getApp(); }
inline int32_t Task::getApp() const { return parent->getId(); }


};
#endif
