#ifndef GMAC_TESTS_COMMON_UTILS_H_
#define GMAC_TESTS_COMMON_UTILS_H_

#include <stdio.h>
#include "config/config.h"

#if defined(HAVE_STDINT)
#include <stdint.h>
#elif defined(_MSC_VER)
typedef unsigned __int8 uint8_t;
typedef __int8 int8_t;
typedef unsigned __int16 uint16_t;
typedef __int16 int16_t;
typedef unsigned __int32 uint32_t;
typedef __int32 int32_t;
typedef unsigned __int64 uint64_t;
typedef __int64 int64_t;
#else
#error "std types not defined!"
#endif

#if !defined(HAVE_LLABS) && defined(_MSC_VER)
#	define llabs _abs64
#endif

/* Timing functions */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	unsigned long sec;
	unsigned long usec;
} gmactime_t;

void getTime(gmactime_t *);

void printTime(gmactime_t *, gmactime_t *, const char *, const char *);

void printAvgTime(gmactime_t *, gmactime_t *, const char *, const char *, unsigned);

void randInit(float *a, size_t size);

void randInitMax(float *a, float maxVal, size_t size);

void valueInit(float *a, float f, size_t size);

#if HAVE_PTHREADS
#include <pthread.h>
typedef pthread_t thread_t;
#elif _MSC_VER
#include <windows.h>
typedef HANDLE thread_t;
#else
#error "No thread support found"
#endif
typedef void*(*thread_routine)(void *);
thread_t thread_create(thread_routine rtn, void *arg);
void thread_wait(thread_t id);


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
	else              *param = def;
}

#include <cmath>

template<typename T>
static
T checkError(const T * orig, const T * calc, uint32_t elems, T (*abs_fn)(T));

template<typename T>
static
void vecAdd(T * c, const T * a, const T * b, uint32_t elems);

#include "utils.ipp"

inline
float
checkError(const float * orig, const float * calc, uint32_t elems)
{
    return checkError<float>(orig, calc, elems, fabsf);
}

inline
double
checkError(const double * orig, const double * calc, uint32_t elems)
{
    return checkError<double>(orig, calc, elems, fabs);
}

inline
long double
checkError(const long double * orig, const long double * calc, uint32_t elems)
{
    return checkError<long double>(orig, calc, elems, fabsl);
}

inline
int
checkError(const int * orig, const int * calc, uint32_t elems)
{
    return checkError<int>(orig, calc, elems, abs);
}

inline
long int
checkError(const long int * orig, const long int * calc, uint32_t elems)
{
    return checkError<long int>(orig, calc, elems, labs);
}

inline
long long int
checkError(const long long int * orig, const long long int * calc, uint32_t elems)
{
    return checkError<long long int>(orig, calc, elems, llabs);
}

#endif

#endif
