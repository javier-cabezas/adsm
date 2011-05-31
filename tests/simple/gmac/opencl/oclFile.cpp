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

const unsigned vecSize = 1024 * 1024;
const unsigned blockSize = 256;

const char *msg = "Done!";

const char *kernel = "\
__kernel void vecSet(__global float *_a, unsigned size, float val)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    _a[i] = val;\
}\
\
__kernel void vecAccum(__global float *_b, __global const float *_a, unsigned size)\
{\
    unsigned i = get_global_id(0);\
    if(i >= size) return;\
\
    _b[i] += _a[i];\
}\
\
__kernel void vecMove(__global float *_a, __global const float *_b, unsigned size)\
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

float error_compute, error_io;

double timeAlloc  = 0.0;
double timeMemset = 0.0;
double timeRun    = 0.0;
double timeCheck  = 0.0;
double timeFree   = 0.0;
double timeWrite  = 0.0;
double timeRead   = 0.0;

void *doTest(void *)
{
    size_t localSize = blockSize;
    size_t globalSize = vecSize / blockSize;
    gmactime_t s, t;
    if(vecSize % blockSize) globalSize++;
    globalSize *= localSize;

    // Alloc & init input data
    getTime(&s);
    if(oclMalloc((void **)&a, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    if(oclMalloc((void **)&b, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    if(oclMalloc((void **)&c, vecSize * sizeof(float)) != oclSuccess)
        CUFATAL();
    getTime(&t);
    timeAlloc += getTimeStamp(t) - getTimeStamp(s);

    getTime(&s);
    oclMemset(a, 0, vecSize * sizeof(float));
    oclMemset(b, 0, vecSize * sizeof(float));
    getTime(&t);
    timeMemset += getTimeStamp(t) - getTimeStamp(s);

    barrier_wait(&ioBefore);

    ocl_kernel kernelSet;

    assert(oclGetKernel("vecSet", &kernelSet) == oclSuccess);

    cl_mem tmp = cl_mem(oclPtr(a));
    assert(oclSetKernelArg(kernelSet, 0, sizeof(cl_mem), &tmp) == oclSuccess);
    assert(oclSetKernelArg(kernelSet, 1, sizeof(vecSize), &vecSize) == oclSuccess);

    ocl_kernel kernelMove;

    assert(oclGetKernel("vecMove", &kernelMove) == oclSuccess);

    tmp = cl_mem(oclPtr(c));
    assert(oclSetKernelArg(kernelMove, 0, sizeof(cl_mem), &tmp) == oclSuccess);
    tmp = cl_mem(oclPtr(a));
    assert(oclSetKernelArg(kernelMove, 1, sizeof(cl_mem), &tmp) == oclSuccess);
    assert(oclSetKernelArg(kernelMove, 2, sizeof(vecSize), &vecSize) == oclSuccess);

    for (int i = 0; i < ITERATIONS; i++) {
        getTime(&s);
        float val = float(i);
        assert(oclSetKernelArg(kernelSet, 2, sizeof(val), &val) == oclSuccess);
        assert(oclCallNDRange(kernelSet, 1, NULL, &globalSize, &localSize) == oclSuccess);

        getTime(&t);
        timeRun += getTimeStamp(t) - getTimeStamp(s);
        barrier_wait(&ioAfter);
        getTime(&s);
        assert(oclCallNDRange(kernelMove, 1, NULL, &globalSize, &localSize) == oclSuccess);
        getTime(&t);
        timeRun += getTimeStamp(t) - getTimeStamp(s);
        barrier_wait(&ioBefore);
    }

    barrier_wait(&ioBefore);

    ocl_kernel kernelAccum;
    assert(oclGetKernel("vecAccum", &kernelAccum) == oclSuccess);

    tmp = cl_mem(oclPtr(b));
    assert(oclSetKernelArg(kernelAccum, 0, sizeof(cl_mem), &tmp) == oclSuccess);
    tmp = cl_mem(oclPtr(a));
    assert(oclSetKernelArg(kernelAccum, 1, sizeof(cl_mem), &tmp) == oclSuccess);
    assert(oclSetKernelArg(kernelAccum, 2, sizeof(vecSize), &vecSize) == oclSuccess);

    for (int i = ITERATIONS - 1; i >= 0; i--) {
        barrier_wait(&ioBefore);
        barrier_wait(&ioAfter);
        getTime(&s);
        assert(oclCallNDRange(kernelAccum, 1, NULL, &globalSize, &localSize) == oclSuccess);
        getTime(&t);
        timeRun += getTimeStamp(t) - getTimeStamp(s);
    }


    error_compute = 0.f;
    getTime(&s);
    for(unsigned i = 0; i < vecSize; i++) {
        error_compute += b[i] - (ITERATIONS - 1)*(ITERATIONS / 2);
    }
    getTime(&t);
    timeCheck += getTimeStamp(t) - getTimeStamp(s);
    getTime(&s);
    oclReleaseKernel(kernelSet);
    oclReleaseKernel(kernelMove);
    oclReleaseKernel(kernelAccum);

    oclFree(a);
    oclFree(b);
    getTime(&t);
    timeFree += getTimeStamp(t) - getTimeStamp(s);

    return &error_compute;
}

static void
setPath(char *name, size_t len, int it)
{
    static const char path_base[] = "_ocl_file_";
    memset(name, '\0', len);
    sprintf(name, "%s%d", path_base, it);
}

static void
writeFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);
    gmactime_t s, t;

    getTime(&s);
    FILE * f = fopen(path, "wb");
    assert(f != NULL);
    assert(fwrite(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
    getTime(&t);
    timeWrite += getTimeStamp(t) - getTimeStamp(s);
}

static void
readFile(float *v, unsigned nmemb, int it)
{
    char path[256];
    setPath(path, 256, it);
    gmactime_t s, t;

    getTime(&s);
    FILE * f = fopen(path, "rb");
    assert(f != NULL);
    assert(fread(v, sizeof(float), nmemb, f) == nmemb);
    fclose(f);
    getTime(&t);
    timeRead += getTimeStamp(t) - getTimeStamp(s);
}

void *doTestIO(void *)
{
    error_io = 0.0f;
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

    return &error_io;
}

int main(int argc, char *argv[])
{
    thread_t tid, tidIO;

    assert(oclCompileSource(kernel) == oclSuccess);

    barrier_init(&ioAfter,2);
    barrier_init(&ioBefore, 2);

    tid = thread_create(doTest, NULL);
    tidIO = thread_create(doTestIO, NULL);

    thread_wait(tid);
    thread_wait(tidIO);

    barrier_destroy(&ioAfter);
    barrier_destroy(&ioBefore);

    fprintf(stdout, "Alloc: %f\n", timeAlloc);
    fprintf(stdout, "Memset: %f\n", timeMemset);
    fprintf(stdout, "Run: %f\n", timeRun);
    fprintf(stdout, "Check: %f\n", timeCheck);
    fprintf(stdout, "Free: %f\n", timeFree);
    fprintf(stdout, "Write: %f\n", timeWrite);
    fprintf(stdout, "Read: %f\n", timeRead);

    return error_io != 0.f;
}
