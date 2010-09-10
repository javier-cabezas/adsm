#include <stdio.h>
#include <gmac.h>

const size_t items = 4;
const size_t size = 128 * 1024;


int main(int argc, char *argv[])
{
	char *ptr[items];

    for(int i = 0; i < items; i++) {
	    assert(gmacMalloc((void **)&ptr[i], size) == gmacSuccess);
    }



    for(int i = 0; i < items; i++) {
	    gmacFree(ptr[i]);
    }

    return 0;
}
