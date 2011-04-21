#include "../utils.h"

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

	out->sec = (unsigned)(tmp / 1000000UL);
	out->usec = (unsigned)(tmp % 1000000UL);
}

thread_t thread_create(thread_routine rtn, void *arg)
{
	thread_t ret = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)rtn, arg, 0, NULL);
	return ret;
}

void thread_wait(thread_t id)
{
	WaitForSingleObject(id, INFINITE);
	CloseHandle(id);
}
