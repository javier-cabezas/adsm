#include "utils.h"


void printTime(struct timeval *start, struct timeval *end, const char *pre, const char *post)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	fprintf(stderr,"%s%f%s", pre, (e - s) / 1e6, post);
}

void printAvgTime(struct timeval *start, struct timeval *end, const char *pre, const char *post, unsigned rounds)
{
	double s, e;
	s = 1e6 * start->tv_sec + (start->tv_usec);
	e = 1e6 * end->tv_sec + (end->tv_usec);
	fprintf(stderr,"%s%f%s", pre, (e - s) / 1e6 / rounds, post);
}

