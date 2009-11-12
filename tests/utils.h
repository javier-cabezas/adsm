#ifndef __TIME_H_
#define __TIME_H_

#include <stdio.h>
#include <sys/time.h>

/* Timing functions */
#ifdef __cplusplus
extern "C" {
#endif
void printTime(struct timeval *, struct timeval *, const char *, const char *);

void printAvgTime(struct timeval *, struct timeval *, const char *, const char *, unsigned);

void randInit(float *a, size_t size);

void randInitMax(float *a, float maxVal, size_t size);

void valueInit(float *a, float f, size_t size);

#ifdef __cplusplus
}
#endif

/* Param functions */
#ifdef __cplusplus

#include <cstdlib>

template<typename T>
void setParam(T *param, const char *str, const T def)
{
	const char *value = getenv(str);
	if(value != NULL) *param = atoi(value);
	if(*param == 0) *param = def;
}

#include <stdint.h>
#include <cmath>

template<typename T>
static
T checkError(const T * orig, const T * calc, uint32_t elems, T (*abs_fn)(T));

template<typename T>
static
void vecAdd(T * c, const T * a, const T * b, uint32_t elems);

#include "utils.ipp"

static
float
checkError(const float * orig, const float * calc, uint32_t elems)
{
    return checkError<float>(orig, calc, elems, fabsf);
}

static
double
checkError(const double * orig, const double * calc, uint32_t elems)
{
    return checkError<double>(orig, calc, elems, fabs);
}

static
long double
checkError(const long double * orig, const long double * calc, uint32_t elems)
{
    return checkError<long double>(orig, calc, elems, fabsl);
}

static
int
checkError(const int * orig, const int * calc, uint32_t elems)
{
    return checkError<int>(orig, calc, elems, abs);
}

static
long int
checkError(const long int * orig, const long int * calc, uint32_t elems)
{
    return checkError<long int>(orig, calc, elems, labs);
}

static
long long int
checkError(const long long int * orig, const long long int * calc, uint32_t elems)
{
    return checkError<long long int>(orig, calc, elems, llabs);
}

#endif

#endif
