#include <windows.h>

#include "trace/Tracer.h"

#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64

namespace __impl { namespace trace {

uint64_t Tracer::timeMark() const
{
	FILETIME ft;
	GetSystemTimeAsFileTime(&ft);

	uint64_t ret = 0;
	ret |= ft.dwHighDateTime;
	ret <<= 32;
	ret |= ft.dwLowDateTime;
	ret -= DELTA_EPOCH_IN_MICROSECS;
	ret /= 10;
	return ret - base_;
}

}}
