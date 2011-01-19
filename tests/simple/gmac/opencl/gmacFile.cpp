#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <gmac/opencl.h>

#include "utils.h"
#include "barrier.h"
#include "debug.h"

#ifdef _MSC_VER
#define VECTORA "inputset\\vectorA"
#define VECTORB "inputset\\vectorB"
#define VECTORC "inputset\\vectorC"
#else
#define VECTORA "inputset/vectorA"
#define VECTORB "inputset/vectorB"
#define VECTORC "inputset/vectorC"
#endif

const size_t vecSize = 1024 * 1024;
const size_t blockSize = 512;

const char *msg = "Done!";

const char *kernel = "\
__kernel void vecSet(__global float *_a, unsigned long size, float val)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    _a[i] = val;\
}\
\
__kernel void vecAccum(__global float *_b, __global const float *_a, unsigned long size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    _b[i] += _a[i];\
}\
\
__kernel void vecMove(__global float *_a, __global const float *_b, unsigned long size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    _a[i] = _b[i];\
}\
";

#define ITERATIONS 250

barrier_t ioAfter;
barrier_t ioBefore;

static void
writeFile(float *v, unsigned nmemb, int it);
static void
readFile(float *v, unsigned nmemb, int it);

float *a, *b, *c;

float error1, error2;

void *doTest(void *)
{
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    // Alloc & init input data
    if(gmacMalloc((void **)&a, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&b, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();
    if(gmacMalloc((void **)&c, vecSize * sizeof(float)) != gmacSuccess)
        CUFATAL();

    gmacMemset(a, 0, vecSize * sizeof(float));
    gmacMemset(b, 0, vecSize * sizeof(float));

    barrier_wait(&ioBefore);

    for (int i = 0; i < ITERATIONS; i++) {
        //vecSet<<<Dg, Db>>>(gmacPtr(a), vecSize, float(i));

        assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
        cl_mem tmp = cl_mem(gmacPtr(a));
        __oclPushArgument(&tmp, sizeof(cl_mem));
        __oclPushArgument(&vecSize, sizeof(vecSize));
        float val = float(i);
        __oclPushArgument(&val, sizeof(val));
        assert(__oclLaunch("vecSet") == gmacSuccess);

        barrier_wait(&ioAfter);
        //vecMove<<<Dg, Db>>>(gmacPtr(c), gmacPtr(a), vecSize);

        assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
        tmp = cl_mem(gmacPtr(c));
        __oclPushArgument(&tmp, sizeof(cl_mem));
        tmp = cl_mem(gmacPtr(a));
        __oclPushArgument(&tmp, sizeof(cl_mem));
        __oclPushArgument(&vecSize, sizeof(vecSize));
        assert(__oclLaunch("vecMove") == gmacSuccess);

        assert(gmacThreadSynchronize() == gmacSuccess);
        barrier_wait(&ioBefore);
    }

    barrier_wait(&ioBefore);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        barrier_wait(&ioBefore);
        barrier_wait(&ioAfter);
        //readFile(a, vecSize, i);
        //vecAccum<<<Dg, Db>>>(gmacPtr(b), gmacPtr(a), vecSize);

        assert(__oclConfigureCall(1, NULL, &globalSize, &localSize) == gmacSuccess);
        cl_mem tmp = cl_mem(gmacPtr(b));
        __oclPushArgument(&tmp, sizeof(cl_mem));
        tmp = cl_mem(gmacPtr(a));
        __oclPushArgument(&tmp, sizeof(cl_mem));
        __oclPushArgument(&vecSize, sizeof(vecSize));
        assert(__oclLaunch("vecAccum") == gmacSuccess);

        gmacThreadSynchronize();
    }


    error1 = 0.f;
    for(unsigned i = 0; i < vecSize; i++) {
        error1 += b[i] - (ITERATIONS - 1)*(ITERATIONS / 2);
    }
    fprintf(stderr,"Error: %f\n", error1);

    gmacFree(a);
    gmacFree(b);

    return &error1;
}

static void
setPath(char *name, size_t len, int it)
{
    static const char path_base[] = "_gmac_file_";
    memset(name, '\0', len);
    sprintf(name, "%s%d", path_base, it);
}

static void
writeFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);

    FILE * f = fopen(path, "wb");
    assert(f != NULL);
    assert(fwrite(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
}

static void
readFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);

    FILE * f = fopen(path, "rb");
    assert(f != NULL);
    assert(fread(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
}

void *doTestIO(void *)
{
    barrier_wait(&ioBefore);

    for (int i = 0; i < ITERATIONS; i++) {
        barrier_wait(&ioAfter);
        barrier_wait(&ioBefore);
        writeFile(c, vecSize, i);
    }

    barrier_wait(&ioBefore);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        barrier_wait(&ioBefore);
        readFile(a, vecSize, i);
        barrier_wait(&ioAfter);
    }

    return &error2;
}

int main(int argc, char *argv[])
{
    thread_t tid, tidIO;

    assert(__oclPrepareCLCode(kernel) == gmacSuccess);

	fprintf(stdout, "Vector: %f\n", 1.0 * vecSize / 1024 / 1024);

    barrier_init(&ioAfter,2);
    barrier_init(&ioBefore, 2);

    tid = thread_create(doTest, NULL);
    tidIO = thread_create(doTestIO, NULL);

    thread_wait(tid);
    thread_wait(tidIO);

    barrier_destroy(&ioAfter);
    barrier_destroy(&ioBefore);

    return error2 != 0.f;
}
