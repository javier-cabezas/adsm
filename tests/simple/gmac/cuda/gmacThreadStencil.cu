#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#include <pthread.h>
#include "barrier.h"

#include "utils.h"
#include "debug.h"

#include "gmacStencilCommon.cu"

const char * nIterStr = "GMAC_NITER";
const unsigned nIterDefault        = 4;

static unsigned nIter = 0;


int main(int argc, char *argv[])
{
	gmactime_t s, t;
	setParam<size_t>(&dimRealElems, dimRealElemsStr, dimRealElemsDefault);
	setParam<unsigned>(&nIter, nIterStr, nIterDefault);

    if (nIter == 0) {
        fprintf(stderr, "Error: nIter should be greater than 0\n");
        abort();
    }

    if (dimRealElems % 32 != 0) {
        fprintf(stderr, "Error: wrong dimension %u\n", unsigned(dimRealElems));
        abort();
    }

    dimElems = dimRealElems + 2 * STENCIL;

    JobDescriptor * descriptors = new JobDescriptor[nIter];
    pthread_t * nThread = new pthread_t[nIter];

    if (nIter > 1) {
        pthread_mutex_init(&mutex, NULL);
        //pthread_barrier_init(&barrier, NULL, nIter);
        barrier_init(&barrier, nIter);
    }

    for(unsigned n = 0; n < nIter; n++) {
        descriptors[n] = JobDescriptor();
        descriptors[n].gpus  = nIter;
        descriptors[n].gpuId = n;

        if (n > 0) {
            descriptors[n].prev = &descriptors[n - 1];
        } else {
            descriptors[n].prev = NULL;
        }

        if (n < nIter - 1) {
            descriptors[n].next = &descriptors[n + 1];
        } else {
            descriptors[n].next = NULL;
        }

        descriptors[n].gpuId = n;
        descriptors[n].dimRealElems = dimRealElems;
        descriptors[n].dimElems     = dimElems;
        descriptors[n].slices       = dimElems / nIter;
	}

	getTime(&t);
	for(unsigned n = 0; n < nIter; n++) {
		pthread_create(&nThread[n], NULL, do_stencil, (void *) &descriptors[n]);
    }

	for(unsigned n = 0; n < nIter; n++) {
		pthread_join(nThread[n], NULL);
	}
	getTime(&s);
	printTime(&t, &s, "Total: ", "\n");

    if (nIter > 1) {
        //pthread_barrier_destroy(&barrier);
        barrier_destroy(&barrier);
    }

    delete descriptors;
    delete nThread;

    return 0;
}
