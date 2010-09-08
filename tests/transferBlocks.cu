#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <sys/time.h>

#include <pthread.h>

#include <cuda.h>
#define CUDA_ERRORS
#include "debug.h"
#include "utils.h"

#define MAXDOUBLE DBL_MAX
#define MINDOUBLE DBL_MIN

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define BANDWIDTH(s, t) ((s) * 8.0 / 1000.0 / (t))

const char *pageLockedStr = "GMAC_PAGE_LOCKED";
const bool pageLockedDefault = false;
bool pageLocked = false;

const char *minSizeStr = "GMAC_MIN";
const size_t minSizeDefault = 4 * 1024;
size_t minSize = 4 * 1024;

const char *maxSizeStr = "GMAC_MAX";
const size_t maxSizeDefault = 64 * 1024 * 1024;
size_t maxSize = 64 * 1024 * 1024;

const char *transferSizeStr = "GMAC_TRANSFER";
const size_t transferSizeDefault = 4 * maxSizeDefault;
size_t transferSize = 64 * 1024 * 1024;


typedef unsigned long long usec_t;
typedef struct {
	double _time, _max, _min, _memcpy, _memcpy_max, _memcpy_min;
} stamp_t;

const int iters = 10;
//const size_t block_elems = 512;

static uint8_t *dev;
static uint8_t *cache;
static uint8_t *cpu;
static uint8_t *cpuTmp;

inline usec_t get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	usec_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
	return tm;
}

enum TransferType {
    TRANSFER_IN,
    TRANSFER_OUT
};

struct param_t {
    TransferType type;
    stamp_t * stamp;
    size_t size;
    size_t totalSize;
};

void fill_cache()
{
    for (int i = 0; i < maxSize; i++) {
        cache[i] = 0;
    }
}

void * transfer(param_t param)
{
    stamp_t * stamp = param.stamp;
	usec_t start, end;

    stamp->_time = 0;
    stamp->_max = MINDOUBLE;
    stamp->_min = MAXDOUBLE;
    stamp->_memcpy = 0;
    stamp->_memcpy_max = MINDOUBLE;
    stamp->_memcpy_min = MAXDOUBLE;

    for(int j = 0; j < iters; j++) {
        if (param.type == TRANSFER_IN) {
            fill_cache();
            for (int c = 0; c < param.totalSize/param.size; c++) {
                start = get_time();
                if(cudaMemcpy(cpu, dev + param.size * c, param.size, cudaMemcpyDeviceToHost) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = get_time();
                stamp->_time += end - start;
                stamp->_min = MIN(stamp->_min, (end - start));
                stamp->_max = MAX(stamp->_max, (end - start));
                if (pageLocked) {
                    start = get_time();
                    memcpy(cpuTmp + param.size * c, cpu, param.size);
                    end = get_time();
                    stamp->_memcpy += end - start;
                    stamp->_memcpy_min = MIN(stamp->_memcpy_min, (end - start));
                    stamp->_memcpy_max = MAX(stamp->_memcpy_max, (end - start));
                }
            }
        } else {
            fill_cache();
            for (int c = 0; c < param.totalSize/param.size; c++) {
                if (pageLocked) {
                    start = get_time();
                    memcpy(cpu, cpuTmp + param.size * c, param.size);
                    end = get_time();
                    stamp->_memcpy += end - start;
                    stamp->_memcpy_min = MIN(stamp->_memcpy_min, (end - start));
                    stamp->_memcpy_max = MAX(stamp->_memcpy_max, (end - start));
                }
                start = get_time();
                if(cudaMemcpy(dev + param.size * c, cpu, param.size, cudaMemcpyHostToDevice) != cudaSuccess)
                    CUFATAL();
                cudaThreadSynchronize();
                end = get_time();
                stamp->_time += end - start;
                stamp->_min = MIN(stamp->_min, (end - start));
                stamp->_max = MAX(stamp->_max, (end - start));
            }
        }
    }
    stamp->_time = stamp->_time / iters;
    stamp->_memcpy = stamp->_memcpy / iters;

    return NULL;
}


int main(int argc, char *argv[])
{
    setParam<bool>(&pageLocked, pageLockedStr, pageLockedDefault);
    setParam<size_t>(&maxSize, maxSizeStr, maxSizeDefault);
    setParam<size_t>(&minSize, minSizeStr, minSizeDefault);
    setParam<size_t>(&transferSize, transferSizeStr, transferSizeDefault);

	stamp_t stamp;
    param_t param;
    param.stamp = &stamp;
    param.totalSize = transferSize;

    // Alloc & init input data
    if(pageLocked) {
        if(cudaHostAlloc((void **) &cpu, maxSize, cudaHostAllocPortable) != cudaSuccess)
            FATAL();
        if((cpuTmp = (uint8_t *) malloc(param.totalSize)) == NULL)
            FATAL();
    } else {
        if((cpu = (uint8_t *) malloc(maxSize)) == NULL)
            FATAL();
    }

	if (cudaMalloc((void **) &dev, param.totalSize) != cudaSuccess)
		CUFATAL();

    if((cache = (uint8_t *) malloc(maxSize)) == NULL)
		CUFATAL();

	// Transfer data
	fprintf(stdout, "#Bytes\tTime\tBwd");
    if (pageLocked) {
	    fprintf(stdout, "\tcuda\tmemcpy");
    }
    /*
	fprintf(stdout, "Min In Bandwidth\tMin Out Bandwidth\t");
	fprintf(stdout, "Max In Bandwidth\tMax Out Bandwidth\n");
    */
	fprintf(stdout, "\n");
	for (int i = minSize; i <= maxSize; i *= 2) {
        param.size = i;
        param.type = TRANSFER_IN;
        transfer(param);
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
        fprintf(stdout, "%f\t%f\t", stamp._time + stamp._memcpy, BANDWIDTH(param.totalSize, stamp._time + stamp._memcpy));
        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t", BANDWIDTH(param.totalSize, stamp._time),
                                        BANDWIDTH(param.totalSize, stamp._memcpy));
        }
        fprintf(stdout, "\n");

#if 0
        param.type = TRANSFER_OUT;
        transfer(param);
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
		fprintf(stdout, "%f\t%f\t", stamp._time, BANDWIDTH(param.totalSize, stamp._time + stamp._memcpy));
        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t", BANDWIDTH(param.totalSize, stamp._time),
                                        BANDWIDTH(param.totalSize, stamp._memcpy));
        }
        fprintf(stdout, "\n");
#endif
	}

    for (int i = minSize; i <= maxSize; i *= 2) {
        param.size = i;
        param.type = TRANSFER_OUT;
        transfer(param);
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
		fprintf(stdout, "%f\t%f\t", stamp._time + stamp._memcpy, BANDWIDTH(param.totalSize, stamp._time + stamp._memcpy));
        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t", BANDWIDTH(param.totalSize, stamp._time),
                                        BANDWIDTH(param.totalSize, stamp._memcpy));
        }
        fprintf(stdout, "\n");

#if 0
        param.type = TRANSFER_OUT;
        transfer(param);
		if (i > 1024 * 1024) fprintf(stdout, "%dMB\t", i / 1024 / 1024);
		else if (i > 1024) fprintf(stdout, "%dKB\t", i / 1024);
		fprintf(stdout, "%f\t%f\t", stamp._time, BANDWIDTH(param.totalSize, stamp._time + stamp._memcpy));
        if (pageLocked) {
            fprintf(stdout, "%f\t%f\t", BANDWIDTH(param.totalSize, stamp._time),
                                        BANDWIDTH(param.totalSize, stamp._memcpy));
        }
        fprintf(stdout, "\n");
#endif
	}

    if(pageLocked) {
        cudaFreeHost(cpu);
    } else {
        free(cpu);
    }

	// Release memory
	cudaFree(dev);
}
