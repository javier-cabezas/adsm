#include <os/loader.h>

#include <debug.h>
#include <paraver.h>
#include <order.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include "mpi_local.h"

SYM(int, __MPI_Sendrecv, void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
SYM(int, __MPI_Send    , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Ssend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Rsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );
SYM(int, __MPI_Bsend   , void *, int, MPI_Datatype, int, int, MPI_Comm );

SYM(int, __MPI_Recv    , void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

void mpiInit(void)
{
	TRACE("Overloading MPI_Sendrecv");
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
    if(__MPI_Sendrecv == NULL) mpiInit();

	// Locate memory regions (if any)
	gmac::Context *sendCtx = manager->owner(sendbuf);
	gmac::Context *recvCtx = manager->owner(recvbuf);

	if (sendCtx == NULL && recvCtx == NULL) {
        return __MPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag, recvbuf, recvcount, recvtype, source, recvtag, comm, status);
    }

    __enterGmac();

    gmac::Context * ctx = gmac::Context::current();
    gmacError_t err;
    int ret;

    size_t bufferSize = ctx->bufferPageLockedSize();
    uint8_t * buffer  = (uint8_t *) ctx->bufferPageLocked();

    size_t bufferUsed = 0;

    void * tmpSend = NULL;
    void * tmpRecv = NULL;

    bool allocSend = false;
    bool allocRecv = false;

    int typebytes;

    if(dest != MPI_PROC_NULL && sendCtx != NULL) {
        size_t sendbytes;

        ret = MPI_Type_size(sendtype, &typebytes);
        if (ret != MPI_SUCCESS) goto exit;
        sendbytes = sendcount * typebytes;

        //! \todo Why is this?
		manager->flush(sendbuf, sendbytes);

        if (bufferSize >= sendbytes) {
            tmpSend  = buffer;
            buffer     += sendbytes;
            bufferUsed += sendbytes;
        } else {
            // Alloc buffer
            err = manager->halloc(ctx, &tmpSend, sendbytes);
            if (err != gmacSuccess) {
                //! \todo Do something
            }
            allocSend = true;
        }

		err = sendCtx->copyToHostAsync(tmpSend,
                                       manager->ptr(sendCtx, sendbuf), sendbytes);
        ASSERT(err == gmacSuccess);
        sendCtx->syncToHost();

        ASSERT(err == gmacSuccess);
	} else {
        tmpSend = sendbuf;
    }

    size_t recvbytes;
    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        ret = MPI_Type_size(recvtype, &typebytes);
        if (ret != MPI_SUCCESS) goto cleanup;
        recvbytes = recvcount * typebytes;

        manager->invalidate(recvbuf, recvbytes);

        if (bufferSize - bufferUsed >= recvbytes) {
            tmpRecv = buffer;
        } else {
            // Alloc buffer
            err = manager->halloc(ctx, &tmpRecv, recvbytes);
            if (err != gmacSuccess) {
                //! \todo Do something
            }
            allocRecv = true;
        }
    } else {
        tmpRecv = recvbuf;
    }

    ret = __MPI_Sendrecv(tmpSend, sendcount, sendtype, dest, sendtag, tmpRecv, recvcount, recvtype, source, recvtag, comm, status);

    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        err = recvCtx->copyToDeviceAsync(manager->ptr(recvCtx, recvbuf),
                                         tmpRecv,
                                         recvbytes);
        ASSERT(err == gmacSuccess);
        err = recvCtx->syncToDevice();
        ASSERT(err == gmacSuccess);
    }

cleanup:
    if (allocSend) {
        // Free temporal buffer
        err = manager->hfree(ctx, tmpSend);
        ASSERT(err == gmacSuccess);
    }

    if (allocRecv) {
        // Free temporal buffer
        err = manager->hfree(ctx, tmpRecv);
        ASSERT(err == gmacSuccess);
    }

exit:
	__exitGmac();

    return ret;
}

int __gmac_MPI_Send( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm,
    int (*func)(void *, int, MPI_Datatype, int, int, MPI_Comm))
{
    if(__MPI_Send == NULL) mpiInit();

	// Locate memory regions (if any)
	gmac::Context *sendCtx = manager->owner(buf);

	if (sendCtx == NULL) {
        return __MPI_Send(buf, count, datatype, dest, tag, comm);
    }

    __enterGmac();

    gmac::Context * ctx = gmac::Context::current();
    gmacError_t err;
    int ret;

    size_t bufferSize = ctx->bufferPageLockedSize();
    uint8_t * buffer  = (uint8_t *) ctx->bufferPageLocked();

    size_t bufferUsed = 0;

    void * tmpSend = NULL;

    bool allocSend = false;

    int typebytes;

    if(dest != MPI_PROC_NULL && sendCtx != NULL) {
        size_t sendbytes;

        ret = MPI_Type_size(datatype, &typebytes);
        if (ret != MPI_SUCCESS) goto exit;
        sendbytes = count * typebytes;

        //! \todo Why is this?
		manager->flush(buf, sendbytes);

        if (bufferSize >= sendbytes) {
            tmpSend  = buffer;
            buffer     += sendbytes;
            bufferUsed += sendbytes;
        } else {
            // Alloc buffer
            err = manager->halloc(ctx, &tmpSend, sendbytes);
            if (err != gmacSuccess) {
                //! \todo Do something
                FATAL("Error in MPI_Send");
            }
            allocSend = true;
        }

		err = sendCtx->copyToHostAsync(tmpSend,
                                       manager->ptr(sendCtx, buf), sendbytes);
        ASSERT(err == gmacSuccess);
        sendCtx->syncToHost();

        ASSERT(err == gmacSuccess);
	} else {
        tmpSend = buf;
    }

    ret = func(tmpSend, count, datatype, dest, tag, comm);

    if (allocSend) {
        // Free temporal buffer
        err = manager->hfree(ctx, tmpSend);
        ASSERT(err == gmacSuccess);
    }

exit:
	__exitGmac();

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
    if(__MPI_Recv == NULL) mpiInit();

	// Locate memory regions (if any)
	gmac::Context *recvCtx = manager->owner(buf);

	if (recvCtx == NULL) {
        return __MPI_Recv(buf, count, datatype, source, tag, comm, status);
    }

    __enterGmac();

    gmac::Context * ctx = gmac::Context::current();
    gmacError_t err;
    int ret;

    size_t bufferSize = ctx->bufferPageLockedSize();
    uint8_t * buffer  = (uint8_t *) ctx->bufferPageLocked();

    size_t bufferUsed = 0;

    void * tmpRecv = NULL;

    bool allocRecv = false;

    int typebytes;

    size_t recvbytes;
    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        ret = MPI_Type_size(datatype, &typebytes);
        if (ret != MPI_SUCCESS) goto cleanup;
        recvbytes = count * typebytes;

        manager->invalidate(buf, recvbytes);

        if (bufferSize - bufferUsed >= recvbytes) {
            tmpRecv = buffer;
        } else {
            // Alloc buffer
            err = manager->halloc(ctx, &tmpRecv, recvbytes);
            if (err != gmacSuccess) {
                //! \todo Do something
                FATAL("Error in MPI_Recv");
            }
            allocRecv = true;
        }
    } else {
        tmpRecv = buf;
    }

    ret = __MPI_Recv(tmpRecv, count, datatype, source, tag, comm, status);

    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        err = recvCtx->copyToDeviceAsync(manager->ptr(recvCtx, buf),
                                         tmpRecv,
                                         recvbytes);
        ASSERT(err == gmacSuccess);
        err = recvCtx->syncToDevice();
        ASSERT(err == gmacSuccess);
    }

cleanup:
    if (allocRecv) {
        // Free temporal buffer
        err = manager->hfree(ctx, tmpRecv);
        ASSERT(err == gmacSuccess);
    }

	__exitGmac();

    return ret;
}
