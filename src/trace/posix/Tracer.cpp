#include "trace/Tracer.h"

#include <sys/time.h>

namespace gmac { namespace trace {
uint64_t Tracer::timeMark() const
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	uint64_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm - base_;
}
}}