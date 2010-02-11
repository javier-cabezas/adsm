#include <debug.h>

#include <gmac/init.h>

#include "Context.h"
#include "Module.h"

#include <cassert>

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
	assert(n != 3);
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
	gmac::Context::current()->lock();
	CUresult r = cuArrayGetDescriptor(&desc, array);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	size_t offset = hOffset * desc.Width + wOffset;
	r = cuMemcpyHtoA(array, offset, src, count);
	gmac::Context::current()->unlock();
	return __getCUDAError(r);
}

cudaError_t __cudaMemcpy2D(CUarray dst, size_t wOffset, size_t hOffset,
		const void *src, size_t spitch, size_t width, size_t height)
{
	TRACE("cudaMemcpy2DToArray (%d %d %d)", spitch, width, height);
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

	gmac::Context::current()->lock();
	CUresult r = cuMemcpy2D(&cuCopy);
	gmac::Context::current()->unlock();
	assert(r == CUDA_SUCCESS);
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
	gmac::Context::current()->lock();
	CUresult r = cuArrayGetDescriptor(&cuDesc, (CUarray)array);
	gmac::Context::current()->unlock();
	if(r != CUDA_SUCCESS) {
		__exitGmac();
		return __getCUDAError(r);
	}
	desc->f = __getCUDAChannelFormatKind(cuDesc.Format);
	__setNumberOfChannels(desc, cuDesc.NumChannels, __getChannelSize(cuDesc.Format));
	TRACE("cudaGetChannelDesc %d %d %d %d %d", desc->x, desc->y, desc->z,
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
	TRACE("cudaMallocArray: %d %d with format 0x%x and %d channels",
			width, height, cuDesc.Format, cuDesc.NumChannels);
	__enterGmac();
	gmac::Context::current()->lock();
	CUresult r = cuArrayCreate((CUarray *)array, &cuDesc);
	gmac::Context::current()->unlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaFreeArray(struct cudaArray *array)
{
	__enterGmac();
	gmac::Context::current()->lock();
	CUresult r = cuArrayDestroy((CUarray)array);
	gmac::Context::current()->unlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	assert(kind == cudaMemcpyHostToDevice);
	__enterGmac();
	cudaError_t ret = __cudaMemcpyToArray((CUarray)dst, wOffset, hOffset, src, count);
	__exitGmac();
	return ret;
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t spitch, size_t width,
		size_t height, enum cudaMemcpyKind kind)
{
	assert(kind == cudaMemcpyHostToDevice);
	__enterGmac();
	cudaError_t ret = __cudaMemcpy2D((CUarray)dst, wOffset, hOffset, src, spitch, width,
				height);
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
    Context * ctx = Context::current();
	const Variable *variable = ctx->constant(symbol);
	assert(variable != NULL);
	CUresult r = CUDA_SUCCESS;
	assert(variable->size() >= (count + offset));
	CUdeviceptr ptr = variable->devPtr() + offset;
	switch(kind) {
		case cudaMemcpyHostToDevice:
			TRACE("cudaMemcpyToSymbol HostToDevice %p to 0x%x (%d bytes)", src, ptr, count);
			ctx->lock();
			r = cuMemcpyHtoD(ptr, src, count);
			ctx->unlock();
			return __getCUDAError(r);
		default:
			abort();
	}
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
	};
}

static inline CUaddress_mode __getAddressMode(cudaTextureAddressMode mode)
{
	switch(mode) {
		case cudaAddressModeWrap:
			return CU_TR_ADDRESS_MODE_WRAP;
		case cudaAddressModeClamp:
			return CU_TR_ADDRESS_MODE_CLAMP;
	};
}

#ifdef __cplusplus
extern "C" {
#endif
using gmac::gpu::Texture;

cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
		const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
    const Texture * texture = Context::current()->texture(texref);
	CUresult r;
	__enterGmac();
	gmac::Context::current()->lock();
	for(int i = 0; i < 3; i++) {
		r = cuTexRefSetAddressMode(texture->texRef(), i, __getAddressMode(texref->addressMode[i]));
		if(r != CUDA_SUCCESS) {
			gmac::Context::current()->unlock();
			__exitGmac();
			return __getCUDAError(r);
		}
	}
	r = cuTexRefSetFlags(texture->texRef(), CU_TRSF_READ_AS_INTEGER);
	if(r != CUDA_SUCCESS) {
		gmac::Context::current()->unlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFilterMode(texture->texRef(), __getFilterMode(texref->filterMode));
	if(r != CUDA_SUCCESS) {
		gmac::Context::current()->unlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetFormat(texture->texRef(),
			__getChannelFormatKind(&texref->channelDesc),
			__getNumberOfChannels(&texref->channelDesc));
	if(r != CUDA_SUCCESS) {
		gmac::Context::current()->unlock();
		__exitGmac();
		return __getCUDAError(r);
	}
	r = cuTexRefSetArray(texture->texRef(), (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);

	gmac::Context::current()->unlock();
	__exitGmac();
	return __getCUDAError(r);
}

cudaError_t cudaUnbindTexture(const struct textureReference *texref)
{
    const Texture * texture = Context::current()->texture(texref);
	__enterGmac();
	gmac::Context::current()->lock();
	CUresult r = cuTexRefDestroy(texture->texRef());
	gmac::Context::current()->unlock();
	__exitGmac();
	return __getCUDAError(r);
}

void CUDARTAPI __cudaTextureFetch(const void *tex, void *index, int integer, void *val)
{
	assert(0);
}

int CUDARTAPI __cudaSynchronizeThreads(void**, void*)
{
	assert(0);
}

void CUDARTAPI __cudaMutexOperation(int lock)
{
	assert(0);
}

#ifdef __cplusplus
}
#endif

