#include "utils.h"

#include <windows.h>

#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64

void getTime(gmactime_t *out)
{
	if(out == NULL) return;
	FILETIME ft;
	GetSystemTimeAsFileTime(&ft);

	unsigned __int64 tmp = 0;
	tmp |= ft.dwHighDateTime;
	tmp <<= 32;
	tmp |= ft.dwLowDateTime;
	tmp -= DELTA_EPOCH_IN_MICROSECS;
	tmp /= 10;

	out->usec = (unsigned)(tmp / 1000000UL);
	out->sec = (unsigned)(tmp % 1000000UL);
}
