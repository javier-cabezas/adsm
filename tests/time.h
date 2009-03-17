#ifndef __TIME_H_
#define __TIME_H_

#include <stdio.h>
#include <sys/time.h>

typedef unsigned long long usec_t;

#define MAX(a, b) ((a) > (b)) ? (a) : (b)
#define MIN(a, b) ((a) < (b)) ? (a) : (b)

#define getTime() get_time()
static inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

#endif
