#ifdef USE_MPI

#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

#include "gmac/init.h"
#include "memory/Manager.h"
#include "core/IOBuffer.h"
#include "core/Process.h"
#include "core/Mode.h"

#include "mpi_local.h"

using __impl::core::IOBuffer;
using __impl::core::Mode;
using __impl::core::Process;

using __impl::memory::Manager;

SYM(int, __MPI_Sendrecv, void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

SYM(int, __MPI_Send    , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Ssend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Rsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Bsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );

SYM(int, __MPI_Recv    , void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

void mpiInit(void)
{
	TRACE(GLOBAL, "Overloading MPI_Sendrecv");
	LOAD_SYM(__MPI_Sendrecv, MPI_Sendrecv);

	LOAD_SYM(__MPI_Send,  MPI_Send);
	LOAD_SYM(__MPI_Ssend, MPI_Ssend);
	LOAD_SYM(__MPI_Rsend, MPI_Rsend);
	LOAD_SYM(__MPI_Bsend, MPI_Bsend);

	LOAD_SYM(__MPI_Recv,  MPI_Recv);
}

int MPI_Sendrecv( void *sendbuf, int sendcount, MPI_Datatype sendtype, 
        int dest, int sendtag, 
        void *recvbuf, int recvcount, MPI_Datatype recvtype, 
        int source, int recvtag, MPI_Comm comm, MPI_Status *status )
{
	if(gmac::inGmac() == 1) return __MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    if(__MPI_Sendrecv == NULL) mpiInit();

    Process &proc = Process::getInstance();
	// Check if GMAC owns any of the buffers to be transferred
	Mode *srcMode = proc.owner(hostptr_t(sendbuf));
	Mode *dstMode = proc.owner(hostptr_t(recvbuf));

	if (srcMode == NULL && dstMode == NULL) {
        return __MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    }

    gmac::enterGmac();

    Mode &mode = Mode::current();
    Manager &manager = Manager::getInstance();

    gmacError_t err;
    int ret, ret2;

    void * tmpSend = NULL;
    void * tmpRecv = NULL;

    bool allocSend = false;
    bool allocRecv = false;

    int typebytes;
    size_t sendbytes = 0;
    size_t recvbytes = 0;
    size_t bufferUsed = 0;

    ret  = MPI_Type_size(sendtype, &typebytes);
    sendbytes = sendcount * typebytes;
    ret2 = MPI_Type_size(recvtype, &typebytes);
    recvbytes = recvcount * typebytes;
    IOBuffer *buffer = mode.createIOBuffer(recvcount + sendcount);

    if (ret  != MPI_SUCCESS) goto exit;
    if (ret2 != MPI_SUCCESS) goto exit;

    ASSERTION(buffer->size() >= size_t(recvcount + sendcount));

    if(dest != MPI_PROC_NULL && srcMode != NULL) {
        // Fast path
        if (buffer->size() >= sendbytes) {
            tmpSend     = buffer->addr();

            err = manager.toIOBuffer(*buffer, hostptr_t(sendbuf), sendbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer->wait();
            ASSERTION(err == gmacSuccess);
        } // Slow path
        else {
            // Alloc buffer
            tmpSend = malloc(sendbytes);
            ASSERTION(tmpSend != NULL);
            allocSend = true;

            err = manager.memcpy(hostptr_t(tmpSend), hostptr_t(sendbuf), sendbytes);
            ASSERTION(err == gmacSuccess);
        }
	} else {
        tmpSend = sendbuf;
    }

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (buffer->size() - bufferUsed >= recvbytes) {
            tmpRecv = buffer->addr() + bufferUsed;
        } else {
            // Alloc buffer
            tmpRecv = malloc(recvbytes);
            ASSERTION(tmpRecv != NULL);

            allocRecv = true;
        }
    } else {
        tmpRecv = recvbuf;
    }

    ret = __MPI_Sendrecv(tmpSend, sendcount, sendtype, dest, sendtag, tmpRecv, recvcount, recvtype, source, recvtag, comm, status);

    if (allocSend) {
        // Free temporal buffer
        free(tmpSend);
    }

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (!allocRecv) {
            err = manager.fromIOBuffer(hostptr_t(recvbuf), *buffer, recvbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer->wait();
            ASSERTION(err == gmacSuccess);
        } else {
            err = manager.memcpy(hostptr_t(recvbuf), hostptr_t(tmpSend), recvbytes);
            ASSERTION(err == gmacSuccess);

            // Free temporal buffer
            free(tmpSend);
        }
    }

    if (buffer != NULL) mode.destroyIOBuffer(buffer);

exit:
	gmac::exitGmac();

    return ret;
}

int __gmac_MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
    int (*func)(void *, int, MPI_Datatype, int, int, MPI_Comm))
{
	if(gmac::inGmac() == 1) return func(buf, count, datatype, dest, tag, comm);
    if(__MPI_Send == NULL) mpiInit();

    Process &proc = Process::getInstance();
	// Check if GMAC owns any of the buffers to be transferred
	Mode *srcMode = proc.owner(hostptr_t(buf));

	if (srcMode == NULL) {
        return __MPI_Send(buf, count, datatype, dest, tag, comm);
    }

    gmac::enterGmac();

    Mode &mode = Mode::current();
    Manager &manager = Manager::getInstance();

    gmacError_t err;
    int ret;

    void * tmpSend = NULL;

    bool allocSend = false;

    int typebytes;
    size_t sendbytes = 0;

    ret = MPI_Type_size(datatype, &typebytes);
    sendbytes = count * typebytes;
    IOBuffer *buffer = mode.createIOBuffer(sendbytes);
    if (ret != MPI_SUCCESS) goto exit;

    if (dest != MPI_PROC_NULL && srcMode != NULL) {
        if (buffer->size() >= sendbytes) {
            tmpSend     = buffer->addr();

            err = manager.toIOBuffer(*buffer, hostptr_t(buf), sendbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer->wait();
            ASSERTION(err == gmacSuccess);
        } else {
            // Alloc buffer
            tmpSend = malloc(sendbytes);
            ASSERTION(tmpSend != NULL);
            allocSend = true;

            err = manager.memcpy(hostptr_t(tmpSend), hostptr_t(buf), sendbytes);
            ASSERTION(err == gmacSuccess);
        }
	} else {
        tmpSend = buf;
    }

    ret = func(tmpSend, count, datatype, dest, tag, comm);

    if (allocSend) {
        // Free temporal buffer
        free(tmpSend);
    }

exit:
	gmac::exitGmac();

    return ret;
}


int MPI_Send ( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Send);
}

int MPI_Ssend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Ssend);
}

int MPI_Rsend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Rsend);
}

int MPI_Bsend( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm )
{
    return __gmac_MPI_Send(buf, count, datatype, dest, tag, comm, __MPI_Bsend);
}

int MPI_Recv( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status )
{
	if(gmac::inGmac() == 1) return __MPI_Recv(buf, count, datatype, source, tag, comm, status);
    if(__MPI_Recv == NULL) mpiInit();

    Process &proc = Process::getInstance();
	// Locate memory regions (if any)
	Mode *dstMode = proc.owner(hostptr_t(buf));

	if (dstMode == NULL) {
        return __MPI_Recv(buf, count, datatype, source, tag, comm, status);
    }

    gmac::enterGmac();

    Mode &mode = Mode::current();
    Manager &manager = Manager::getInstance();

    gmacError_t err;
    int ret;

    void * tmpRecv = NULL;

    bool allocRecv = false;

    int typebytes;
    size_t recvbytes = 0;

    ret = MPI_Type_size(datatype, &typebytes);
    recvbytes = count * typebytes;
    IOBuffer *buffer = mode.createIOBuffer(recvbytes);
    if (ret != MPI_SUCCESS) goto exit;

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (buffer->size() >= recvbytes) {
            tmpRecv = buffer->addr();
        } else {
            // Alloc buffer
            tmpRecv = malloc(recvbytes);
            ASSERTION(tmpRecv != NULL);
            allocRecv = true;
        }
    } else {
        tmpRecv = buf;
    }

    ret = __MPI_Recv(tmpRecv, count, datatype, source, tag, comm, status);

    if(source != MPI_PROC_NULL && dstMode != NULL) {
        if (!allocRecv) {
            err = manager.fromIOBuffer(hostptr_t(buf), *buffer, recvbytes);
            ASSERTION(err == gmacSuccess);
            err = buffer->wait();
            ASSERTION(err == gmacSuccess);
        } else {
            err = manager.memcpy(hostptr_t(buf), hostptr_t(tmpRecv), recvbytes);
            ASSERTION(err == gmacSuccess);

            // Free temporal buffer
            free(tmpRecv);
        }
    }

    if (allocRecv) {
        // Free temporal buffer
        free(tmpRecv);
    }

exit:
	gmac::exitGmac();

    return ret;
}

#endif
