#ifndef __TIME_H_
#define __TIME_H_

#include <stdio.h>
#include <sys/time.h>

/* Timing functions */
#ifdef __cplusplus
extern "C" {
#endif
void printTime(struct timeval *, struct timeval *, const char *, const char *);

#ifdef __cplusplus
}
#endif

/* Param functions */
#ifdef __cplusplus
template<typename T>
void setParam(T *param, const char *str, const T def)
{
	const char *value = getenv(str);
	if(value != NULL) *param = atoi(value);
	if(*param == 0) *param = def;
}
#endif


#endif
