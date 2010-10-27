#include "utils.h"

#include <sys/time.h>


void getTime(gmactime_t *out)
{
	if(out == NULL) return;
    struct timeval tv;
    if(gettimeofday(&tv, NULL) < 0) return;
    out->usec = tv.tv_usec;
    out->sec = tv.tv_sec;
}
