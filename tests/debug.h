#ifndef __DEBUG_H_
#define __DEBUG_H_

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include <cuda.h>

#define FATAL()	\
	do {	\
		fprintf(stderr,"[%s:%d] : %s\n", __FILE__, __LINE__, \
			strerror(errno));	\
		exit(-1);	\
	} while(0)

#define CUFATAL()	\
	do {	\
		fprintf(stderr,"[%s:%d] : %s\n", __FILE__, __LINE__, \
			cudaGetErrorString(cudaGetLastError()));	\
		exit(-1);	\
	} while(0)

#endif
