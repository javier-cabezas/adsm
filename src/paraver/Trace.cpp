#include "Trace.h"

#include <strings.h>
#include <assert.h>

#include <iomanip>

namespace paraver {

void Trace::buildApp(std::ifstream &in)
{
	uint32_t id, nTasks;
	in.read((char *)&id, sizeof(id));
	in.read((char *)&nTasks, sizeof(nTasks));

	std::ostringstream name;
	name << "App" << id;
	apps.push_back(new Application(id, name.str()));

	for(uint32_t i = 0; i < nTasks; i++) {
		uint32_t nThreads;
		in.read((char *)&id, sizeof(id));
		in.read((char *)&nThreads, sizeof(nThreads));
		Task *task = apps.back()->addTask(id);
		for(uint32_t j = 0; j < nThreads; j++) {
			uint32_t dummy;
			in.read((char *)&id, sizeof(id));
			in.read((char *)&dummy, sizeof(dummy));
			task->__addThread(id);
		}
	}
}

void Trace::read(const char *fileName)
{
	std::ifstream in;
	in.open(fileName, std::ios::binary);
	// Read records from file
	Record *record = NULL;
	while((record = Record::read(in)) != NULL) {
		records.push_back(record);
		endTime = (endTime > record->getEndTime()) ? endTime : record->getEndTime();
	} 

	// Sort the records
	records.sort(RecordPredicate());

	// Read header
	uint32_t nApps;
	in.read((char *)&nApps, sizeof(nApps));
	for(uint32_t i = 0; i < nApps; i++) buildApp(in);
}

void Trace::write()
{
	std::list<Application *>::iterator app;
	for(app = apps.begin(); app != apps.end(); app++) {
		(*app)->end(of, getTimeStamp());
	}

	Record::end(of);
	uint32_t size = apps.size();
	of.write((char *)&size, sizeof(size));
	for(app = apps.begin(); app != apps.end(); app++) {
		(*app)->write(of);
	}

	of.close();
}

std::ostream &operator<<(std::ostream &of, const Trace &trace)
{
	time_t timep=time(NULL);
	struct tm *t=localtime(&timep);
	int32_t year=(t->tm_year<100)?t->tm_year:t->tm_year-100;
	// Print the file header: date/time
	of << "#Paraver(";
	of << std::setw(2) << std::setfill('0') << t->tm_mday << "/";
	of << std::setw(2) << std::setfill('0') << t->tm_mon << "/";
	of << std::setw(2) << std::setfill('0') << year;
	of << " at " << std::setw(2) << std::setfill('0') << t->tm_hour << ":";
	of << std::setw(2) << std::setfill('0') << t->tm_min << ")";
	of << ":" << trace.endTime; 

	// Without resource mode
	of << ":" << 0;

	// Print # of applications and tasks and threads per task
	of << ":" << trace.apps.size();
	std::list<Application *>::const_iterator app;
	for(app = trace.apps.begin(); app != trace.apps.end(); app++)
		of << ":" << *(*app);
	of << std::endl;

	std::list<Record *>::const_iterator i;
	for(i = trace.records.begin(); i != trace.records.end(); i++)
		of << *(*i);

}

};
