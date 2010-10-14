#include "utils.h"

void printTime(struct timeval *start, struct timeval *end, const char *pre, const char *post)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	printf("%s%f%s", pre, (e - s) / 1e6, post);
}

void printAvgTime(struct timeval *start, struct timeval *end, const char *pre, const char *post, unsigned rounds)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	printf("%s%f%s", pre, (e - s) / 1e6 / rounds, post);
}

void randInit(float *a, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 1.0 * rand();
	}
}

void randInitMax(float *a, float maxVal, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = 1.f * (rand() % int(maxVal));
	}
}

void valueInit(float *a, float v, size_t size)
{
	for(unsigned i = 0; i < size; i++) {
		a[i] = v;
	}
}

void utils_init() __attribute__ ((constructor));

#include <ctime>
void utils_init()
{
    srand(time(NULL));
}
