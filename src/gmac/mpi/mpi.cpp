#include <os/loader.h>

#include <debug.h>
#include <paraver.h>
#include <order.h>

#include <init.h>
#include <memory/Manager.h>
#include <kernel/Context.h>

#include "mpi_local.h"

SYM(size_t, __MPI_Sendrecv, void *, int, MPI_Datatype, int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);

void mpiInit(void)
{
	TRACE("Overloading MPI_Sendrecv");
	LOAD_SYM(__MPI_Sendrecv, MPI_Sendrecv);
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

    if(dest != MPI_PROC_NULL && sendCtx != NULL) {
        //! \todo Why is this?
		manager->flush(sendbuf, sendcount);

        if (int(bufferSize) >= sendcount) {
            tmpSend  = buffer;
            buffer     += sendcount;
            bufferUsed += sendcount;
        } else {
            // Alloc buffer
            //tmpSend     = ;
            allocSend = true;
        }

		err = sendCtx->copyToHostAsync(tmpSend,
                                       manager->ptr(sendCtx, sendbuf), sendcount);
        ASSERT(err == gmacSuccess);
        sendCtx->syncToHost();

        ASSERT(err == gmacSuccess);
	} else {
        tmpSend = sendbuf;
    }

    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        manager->invalidate(recvbuf, recvcount);

        if (int(bufferSize - bufferUsed) >= recvcount) {
            tmpRecv = buffer;
        } else {
            // Alloc buffer
            allocRecv = true;
        }
    } else {
        tmpRecv = recvbuf;
    }

    ret = __MPI_Sendrecv(tmpSend, sendcount, sendtype, dest, sendtag, tmpRecv, recvcount, recvtype, source, recvtag, comm, status);

    if(source != MPI_PROC_NULL && recvCtx != NULL) {
        err = recvCtx->copyToDeviceAsync(manager->ptr(recvCtx, recvbuf),
                                         tmpRecv,
                                         recvcount);
        ASSERT(err == gmacSuccess);
        err = recvCtx->syncToDevice();
        ASSERT(err == gmacSuccess);
    }

    if (allocSend) {
        // Free temporal buffer
    }

    if (allocRecv) {
        // Free temporal buffer
    }

	__exitGmac();

    return ret;
}
