#ifndef __DEBUG_H_
#define __DEBUG_H_

#include <stdio.h>
#include <string.h>
#include <errno.h>

#define FATAL(fmt, ...)	\
	do {	\
		fprintf(stderr,"FATAL [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);	\
		exit(-1);	\
	} while(0)

#ifdef DEBUG
#define TRACE(fmt, ...)	\
	do {	\
		fprintf(stderr,"TRACE [%s:%d] " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__);	\
	} while(0)
#else
#define TRACE(fmt, ...)
#endif


#endif
