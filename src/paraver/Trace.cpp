#include "Trace.h"

#include <strings.h>
#include <assert.h>

namespace paraver {

void Trace::buildNode(std::ifstream &in)
{
	uint32_t id, nCores;
	in.read((char *)&id, sizeof(id));
	in.read((char *)&nCores, sizeof(nCores));

	std::ostringstream name;
	name << "Node" << id;
	nodes.push_back(new Node(id, name.str()));

	// Create cores
	for(uint32_t i = 0; i < nCores; i++) {
		uint32_t null;
		in.read((char *)&id, sizeof(id));
		in.read((char *)&null, sizeof(null));	// Must be 0
		assert(null == 0);
		nodes.back()->addCpu(id);
	}
}

void Trace::buildApp(std::ifstream &in)
{
	uint32_t id, nTasks;
	in.read((char *)&id, sizeof(id));
	in.read((char *)&nTasks, sizeof(nTasks));

	std::ostringstream name;
	name << "App" << id;
	apps.push_back(new Application(id, name.str()));

	for(uint32_t i = 0; i < nTasks; i++) {
		uint32_t node, nThreads;
		in.read((char *)&node, sizeof(node));
		assert(node < nodes.size());
		in.read((char *)&id, sizeof(id));
		in.read((char *)&nThreads, sizeof(nThreads));
		apps.back()->addTask(nodes[node], id);
	}
}

void Trace::read(const char *fileName)
{
	std::ifstream in;
	in.open(fileName, std::ios::in);
	// Read records from file
	do { records.push_back(Record::read(in)); } while(records.back() != NULL);

	// Sort the records
	records.sort(RecordPredicate());

	// Read header
	uint32_t nNodes;
	in.read((char *)&nNodes, sizeof(nNodes));
	for(uint32_t i = 0; i < nNodes; i++) buildNode(in);

	uint32_t nApps;
	in.read((char *)&nApps, sizeof(nApps));
	for(uint32_t i = 0; i < nApps; i++) buildApp(in);
}

void Trace::write()
{
	Record::end(of);
	uint32_t size = nodes.size();
	of.write((char *)&size, sizeof(size));
	std::vector<Node *>::iterator node;
	for(node=nodes.begin(); node!=nodes.end(); node++) {
		(*node)->write(of);
	}
	size = apps.size();
	of.write((char *)&size, sizeof(size));
	std::list<Application *>::iterator app;
	for(app = apps.begin(); app != apps.end(); app++) {
		(*app)->write(of);
	}

	of.close();
}

std::ofstream &operator<<(std::ofstream &of, const Trace &trace)
{
	time_t timep=time(NULL);
	struct tm *t=localtime(&timep);
	int32_t year=(t->tm_year<100)?t->tm_year:t->tm_year-100;
	// Print the file header: date/time
	of << "#Trace(" << t->tm_mday << "/" << t->tm_mon << "/" << year;
	of << " at " << t->tm_hour << ":" << t->tm_min << ")";
	// TODO: print the total time
	of << ":" << trace.endTime; 

	// Print system configuration (Nodes and CPU per node)
	of << ":" << trace.nodes.size() << "(";
	std::vector<Node *>::const_iterator node;
	for(node = trace.nodes.begin(); node != trace.nodes.end(); node++) {
		if(node != trace.nodes.begin()) of << ",";
		of << (*node)->size();
	}
	of << ")";

	// Print # of applications and tasks and threads per task
	of << ":" << trace.apps.size();
	std::list<Application *>::const_iterator app;
	for(app = trace.apps.begin(); app != trace.apps.end(); app++)
		of << ":" << *(app);
	of << std::endl;
}

};
