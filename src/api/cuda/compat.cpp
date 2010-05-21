#include <gmac/init.h>

#include <memory/Manager.h>

#include "Context.h"
#include "Module.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <string>
#include <list>

static inline size_t __getChannelSize(CUarray_format format)
{
	switch(format) {
		case CU_AD_FORMAT_SIGNED_INT8:
		case CU_AD_FORMAT_UNSIGNED_INT8:
			return 8;
		case CU_AD_FORMAT_SIGNED_INT16:
		case CU_AD_FORMAT_UNSIGNED_INT16:
		case CU_AD_FORMAT_HALF:
			return 16;
		case CU_AD_FORMAT_SIGNED_INT32:
		case CU_AD_FORMAT_UNSIGNED_INT32:
		case CU_AD_FORMAT_FLOAT:
			return 32;
	}
	return 0;
}

static inline CUarray_format __getChannelFormatKind(const struct cudaChannelFormatDesc *desc)
{
	int size = desc->x;
	switch(desc->f) {
		case cudaChannelFormatKindSigned:
			if(size == 8) return CU_AD_FORMAT_SIGNED_INT8;
			if(size == 16) return CU_AD_FORMAT_SIGNED_INT16;
			if(size == 32) return CU_AD_FORMAT_SIGNED_INT32;
		case cudaChannelFormatKindUnsigned:
			if(size == 8) return CU_AD_FORMAT_UNSIGNED_INT8;
			if(size == 16) return CU_AD_FORMAT_UNSIGNED_INT16;
			if(size == 32) return CU_AD_FORMAT_UNSIGNED_INT32;
		case cudaChannelFormatKindFloat:
			if(size == 16) return CU_AD_FORMAT_HALF;
			if(size == 32) return CU_AD_FORMAT_FLOAT;
		case cudaChannelFormatKindNone:
            break;
	};
	return CU_AD_FORMAT_UNSIGNED_INT32;
}

static inline unsigned int __getNumberOfChannels(const struct cudaChannelFormatDesc *desc)
{
	unsigned int n = 0;
	if(desc->x != 0) n++;
	if(desc->y != 0) n++;
	if(desc->z != 0) n++;
	if(desc->w != 0) n++;
	gmac::util::Logger::Assertion(n != 3);
	return n;
}

static inline void __setNumberOfChannels(struct cudaChannelFormatDesc *desc, unsigned int channels, int s)
{
	desc->x = desc->y = desc->z = desc->w = 0;
	if(channels >= 1) desc->x = s;
	if(channels >= 2) desc->y = s;
	if(channels >= 4) { desc->z = desc->w = s; }
}



static inline cudaChannelFormatKind __getCUDAChannelFormatKind(CUarray_format format)
{
	switch(format) {
		case CU_AD_FORMAT_UNSIGNED_INT8:
		case CU_AD_FORMAT_UNSIGNED_INT16:
		case CU_AD_FORMAT_UNSIGNED_INT32:
			return cudaChannelFormatKindUnsigned;
		case CU_AD_FORMAT_SIGNED_INT8:
		case CU_AD_FORMAT_SIGNED_INT16:
		case CU_AD_FORMAT_SIGNED_INT32:
			return cudaChannelFormatKindSigned;
		case CU_AD_FORMAT_HALF:
		case CU_AD_FORMAT_FLOAT:
			return cudaChannelFormatKindFloat;
	};
	return cudaChannelFormatKindSigned;
}


static inline cudaError_t __getCUDAError(CUresult r)
{
	switch(r) {
		case CUDA_SUCCESS:
			return cudaSuccess;
		case CUDA_ERROR_OUT_OF_MEMORY:
			return cudaErrorMemoryAllocation;
		case CUDA_ERROR_NOT_INITIALIZED:
		case CUDA_ERROR_DEINITIALIZED:
			return cudaErrorInitializationError;
        case CUDA_ERROR_INVALID_VALUE:
            return cudaErrorInvalidValue;
        case CUDA_ERROR_INVALID_DEVICE:
            return cudaErrorInvalidDevice;
        case CUDA_ERROR_MAP_FAILED:
            return cudaErrorMapBufferObjectFailed;
        case CUDA_ERROR_UNMAP_FAILED:
            return cudaErrorUnmapBufferObjectFailed;
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return cudaErrorLaunchTimeout;
        case CUDA_ERROR_LAUNCH_FAILED:
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return cudaErrorLaunchFailure;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return cudaErrorLaunchOutOfResources;
        case CUDA_ERROR_NO_DEVICE:
#if CUDART_VERSION >= 2020
            return cudaErrorNoDevice;
#endif
#if CUDART_VERSION >= 3000
#if LINUX
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return cudaErrorECCUncorrectable;
#endif
        case CUDA_ERROR_POINTER_IS_64BIT:
        case CUDA_ERROR_SIZE_IS_64BIT:
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
#endif
        case CUDA_ERROR_ARRAY_IS_MAPPED:
        case CUDA_ERROR_ALREADY_MAPPED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_INVALID_IMAGE:
        case CUDA_ERROR_INVALID_SOURCE:
        case CUDA_ERROR_FILE_NOT_FOUND:
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
        case CUDA_ERROR_ALREADY_ACQUIRED:
        case CUDA_ERROR_NOT_FOUND:
        case CUDA_ERROR_NOT_READY:
        case CUDA_ERROR_NOT_MAPPED:
        case CUDA_ERROR_UNKNOWN:
            break;
	};
	return cudaErrorUnknown;
}

static inline CUmemorytype __getMemoryFrom(cudaMemcpyKind kind)
{
	switch(kind) {
		case cudaMemcpyHostToHost:
		case cudaMemcpyHostToDevice:
			return CU_MEMORYTYPE_HOST;
		case cudaMemcpyDeviceToHost:
		case cudaMemcpyDeviceToDevice:
			return CU_MEMORYTYPE_DEVICE;
	}
}

static inline CUmemorytype __getMemoryTo(cudaMemcpyKind kind)
{
	switch(kind) {
		case cudaMemcpyHostToHost:
		case cudaMemcpyDeviceToHost:
			return CU_MEMORYTYPE_HOST;
		case cudaMemcpyHostToDevice:
		case cudaMemcpyDeviceToDevice:
			return CU_MEMORYTYPE_DEVICE;
	}
}

cudaError_t __cudaMemcpyToArray(CUarray array, size_t wOffset,
		size_t hOffset, const void *src, size_t count)
{
	CUDA_ARRAY_DESCRIPTOR desc;
    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuArrayGetDescriptor(&desc, array);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	size_t offset = hOffset * desc.Width + wOffset;
	r = cuMemcpyHtoA(array, offset, src, count);
	ctx->popUnlock();
	return __getCUDAError(r);
}

cudaError_t __cudaMemcpy2D(CUarray dst, size_t wOffset, size_t hOffset,
		const void *src, size_t spitch, size_t width, size_t height)
{
	gmac::util::Logger::Trace("cudaMemcpy2DToArray (%zd %zd %zd)", spitch, width, height);
	CUDA_MEMCPY2D cuCopy;
	memset(&cuCopy, 0, sizeof(cuCopy));

	cuCopy.srcMemoryType = CU_MEMORYTYPE_HOST;
	cuCopy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cuCopy.srcHost = src;
	cuCopy.dstArray = (CUarray)dst;

	cuCopy.srcPitch = spitch;
	cuCopy.WidthInBytes = width;
	cuCopy.Height= height;

	cuCopy.srcXInBytes = wOffset;
	cuCopy.srcY = hOffset;

    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuMemcpy2D(&cuCopy);
	ctx->popUnlock();
	gmac::util::Logger::Assertion(r == CUDA_SUCCESS);
	return __getCUDAError(r);

}

cudaError_t __cudaInternalMemcpy2D(CUarray dst, size_t wOffset, size_t hOffset,
		CUdeviceptr src, size_t spitch, size_t width, size_t height)
{
	gmac::util::Logger::Trace("cudaMemcpy2DToArray (%zd %zd %zd)", spitch, width, height);
	CUDA_MEMCPY2D cuCopy;
	memset(&cuCopy, 0, sizeof(cuCopy));

	cuCopy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	cuCopy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	cuCopy.srcDevice = src;
	cuCopy.dstArray = dst;

	cuCopy.srcPitch = spitch;
	cuCopy.WidthInBytes = width;
	cuCopy.Height= height;

	cuCopy.srcXInBytes = wOffset;
	cuCopy.srcY = hOffset;

    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuMemcpy2DUnaligned(&cuCopy);
	ctx->popUnlock();
	gmac::util::Logger::Assertion(r == CUDA_SUCCESS);
	return __getCUDAError(r);

}

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Channel related functions

struct cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z,
		int w, enum cudaChannelFormatKind kind)
{
	struct cudaChannelFormatDesc desc;
	desc.x = x; desc.y = y; desc.z = z; desc.w = w;
	desc.f = kind;
	return desc;
}

cudaError_t cudaGetChannelDesc(struct cudaChannelFormatDesc *desc,
		const struct cudaArray *array)
{
	__enterGmac();
	CUDA_ARRAY_DESCRIPTOR cuDesc;
    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuArrayGetDescriptor(&cuDesc, (CUarray)array);
	ctx->popUnlock();
	if(r != CUDA_SUCCESS) {
		__exitGmac();
		return __getCUDAError(r);
	}
	desc->f = __getCUDAChannelFormatKind(cuDesc.Format);
	__setNumberOfChannels(desc, cuDesc.NumChannels, __getChannelSize(cuDesc.Format));
	gmac::util::Logger::Trace("cudaGetChannelDesc %d %d %d %d %d", desc->x, desc->y, desc->z,
		desc->w, desc->f);
	__exitGmac();
	return cudaSuccess;
}

// CUDA Array related functions

cudaError_t cudaMallocArray(struct cudaArray **array,
		const struct cudaChannelFormatDesc *desc, size_t width,
		size_t height)
{
	CUDA_ARRAY_DESCRIPTOR cuDesc;
	cuDesc.Width = width; cuDesc.Height = height;
	cuDesc.Format = __getChannelFormatKind(desc);
	cuDesc.NumChannels = __getNumberOfChannels(desc);
	gmac::util::Logger::Trace("cudaMallocArray: %zd %zd with format 0x%x and %u channels",
			width, height, cuDesc.Format, cuDesc.NumChannels);
	__enterGmac();
    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuArrayCreate((CUarray *)array, &cuDesc);
	ctx->popUnlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaFreeArray(struct cudaArray *array)
{
	__enterGmac();
    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuArrayDestroy((CUarray)array);
	ctx->popUnlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	gmac::util::Logger::Assertion(kind == cudaMemcpyHostToDevice);
	__enterGmac();
	cudaError_t ret = __cudaMemcpyToArray((CUarray)dst, wOffset, hOffset, src, count);
	__exitGmac();
	return ret;
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t spitch, size_t width,
		size_t height, enum cudaMemcpyKind kind)
{
	gmac::util::Logger::Assertion(kind == cudaMemcpyHostToDevice);
	__enterGmac();
	cudaError_t ret = cudaSuccess;
    gmac::gpu::Context *ctx = dynamic_cast<gmac::gpu::Context *>(manager->owner(src));
    if(ctx == NULL) {
        __cudaMemcpy2D((CUarray)dst, wOffset, hOffset, src, spitch, width,
				height);
    }
    else {
        __cudaInternalMemcpy2D((CUarray)dst, wOffset, hOffset, ctx->gpuAddr(src), spitch, width,
				height);
    }
	__exitGmac();
	return ret;
}

#ifdef __cplusplus
}
#endif


// Functions related to constant memory

#ifdef __cplusplus
extern "C" {
#endif

using gmac::gpu::Context;
using gmac::gpu::Module;
using gmac::gpu::Variable;

cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind)
{
	__enterGmac();
	cudaError_t ret = cudaSuccess;
    Context * ctx = Context::current();
	const Variable *variable = ctx->constant(symbol);
	gmac::util::Logger::Assertion(variable != NULL);
	CUresult r = CUDA_SUCCESS;
	gmac::util::Logger::Assertion(variable->size() >= (count + offset));
	CUdeviceptr ptr = variable->devPtr() + offset;
	switch(kind) {
		case cudaMemcpyHostToDevice:
			gmac::util::Logger::Trace("cudaMemcpyToSymbol HostToDevice %p to 0x%x (%zd bytes)", src, ptr, count);
			ctx->pushLock();
			r = cuMemcpyHtoD(ptr, src, count);
			ctx->popUnlock();
			ret = __getCUDAError(r);
            break;

		default:
			abort();
	}
    __exitGmac();
    return ret;
}

#ifdef __cplusplus
}
#endif


// CUDA Texture related functions

static inline CUfilter_mode __getFilterMode(cudaTextureFilterMode mode)
{
	switch(mode) {
		case cudaFilterModePoint:
			return CU_TR_FILTER_MODE_POINT;
		case cudaFilterModeLinear:
			return CU_TR_FILTER_MODE_LINEAR;
		default:
			return CU_TR_FILTER_MODE_LINEAR;
	};
}

static inline CUaddress_mode __getAddressMode(cudaTextureAddressMode mode)
{
	switch(mode) {
		case cudaAddressModeWrap:
			return CU_TR_ADDRESS_MODE_WRAP;
		case cudaAddressModeClamp:
			return CU_TR_ADDRESS_MODE_CLAMP;
        default:
			return CU_TR_ADDRESS_MODE_WRAP;
	};
}

#ifdef __cplusplus
extern "C" {
#endif
using gmac::gpu::Texture;

cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
		const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
	__enterGmac();
    Context * ctx = Context::current();
	CUresult r;
    const Texture * texture = ctx->texture(texref);
    ctx->pushLock();
	for(int i = 0; i < 3; i++) {
		r = cuTexRefSetAddressMode(texture->texRef(), i, __getAddressMode(texref->addressMode[i]));
		if(r != CUDA_SUCCESS) {
			ctx->popUnlock();
			__exitGmac();
			return __getCUDAError(r);
		}
	}
	r = cuTexRefSetFlags(texture->texRef(), CU_TRSF_READ_AS_INTEGER);
	if(r != CUDA_SUCCESS) {
		ctx->popUnlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFilterMode(texture->texRef(), __getFilterMode(texref->filterMode));
	if(r != CUDA_SUCCESS) {
		ctx->popUnlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFormat(texture->texRef(),
			__getChannelFormatKind(&texref->channelDesc),
			__getNumberOfChannels(&texref->channelDesc));
	if(r != CUDA_SUCCESS) {
		ctx->popUnlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetArray(texture->texRef(), (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);

	ctx->popUnlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaUnbindTexture(const struct textureReference *texref)
{
	__enterGmac();
    const Texture * texture = Context::current()->texture(texref);
    gmac::Context * ctx = gmac::Context::current();
	ctx->pushLock();
	CUresult r = cuTexRefDestroy(texture->texRef());
	ctx->popUnlock();
	__exitGmac();
	return __getCUDAError(r);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val)
{
	gmac::util::Logger::Assertion(0);
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
	gmac::util::Logger::Assertion(0);
    return 0;
}

void CUDARTAPI __cudaMutexOperation(int lock)
{
	gmac::util::Logger::Assertion(0);
}


// Events and other stuff needed by CUDA Wrapper
cudaError_t cudaEventCreate(cudaEvent_t *event)
{
#if CUDART_VERSION >= 2020
    CUresult ret = cuEventCreate((CUevent *)event, CU_EVENT_DEFAULT);
#else
    CUresult ret = cuEventCreate((CUevent *)event, 0);
#endif
    return __getCUDAError(ret);
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    CUresult ret = cuEventDestroy((CUevent) event);
    return __getCUDAError(ret);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    CUresult ret = cuEventElapsedTime(ms, (CUevent)start, (CUevent)end);
    return __getCUDAError(ret);
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    CUresult ret = cuEventQuery((CUevent) event);
    return __getCUDAError(ret);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    CUresult ret = cuEventRecord((CUevent) event, gmac::gpu::Context::stream());
    return __getCUDAError(ret);
}



#ifdef __cplusplus
}
#endif

