#include "api.h"
#include "driver.h"

#include <config.h>
#include <debug.h>

#include <assert.h>

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
	__gmacError(r);
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
	CUresult r = cuArrayGetDescriptor(&desc, array);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	size_t offset = hOffset * desc.Width + wOffset;
	r = cuMemcpyHtoA(array, offset, src, count);
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

	CUresult r = cuMemcpy2D(&cuCopy);
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
	CUDA_ARRAY_DESCRIPTOR cuDesc;
	CUresult r = cuArrayGetDescriptor(&cuDesc, (CUarray)array);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	desc->f = __getCUDAChannelFormatKind(cuDesc.Format);
	__setNumberOfChannels(desc, cuDesc.NumChannels, __getChannelSize(cuDesc.Format));
	TRACE("cudaGetChannelDesc %d %d %d %d %d", desc->x, desc->y, desc->z,
		desc->w, desc->f);
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
	CUresult r = cuArrayCreate((CUarray *)array, &cuDesc);
	return __getCUDAError(r);
}

cudaError_t cudaFreeArray(struct cudaArray *array)
{
	CUresult r = cuArrayDestroy((CUarray)array);
	return __getCUDAError(r);
}

cudaError_t cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t count,
		enum cudaMemcpyKind kind)
{
	assert(kind == cudaMemcpyHostToDevice);
	return __cudaMemcpyToArray((CUarray)dst, wOffset, hOffset, src, count);
}

cudaError_t cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset,
		size_t hOffset, const void *src, size_t spitch, size_t width,
		size_t height, enum cudaMemcpyKind kind)
{
	assert(kind == cudaMemcpyHostToDevice);
	return __cudaMemcpy2D((CUarray)dst, wOffset, hOffset, src, spitch, width,
				height);
}

#ifdef __cplusplus
}
#endif


// Functions related to constant memory

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaMemcpyToSymbol(const char *symbol, const void *src, size_t count,
		size_t offset, enum cudaMemcpyKind kind)
{
	VariableMap::const_iterator variable = varMap.find(symbol);
	assert(variable != varMap.end());
	assert(variable->second.constant == true);
	CUresult r = CUDA_SUCCESS;
	assert(variable->second.size >= (count + offset));
	CUdeviceptr ptr = variable->second.ptr + offset;
	switch(kind) {
		case cudaMemcpyHostToDevice:
			TRACE("cudaMemcpyToSymbol HostToDevice %p to 0x%x (%d bytes)", src, ptr, count);
			r = cuMemcpyHtoD(ptr, src, count);
			return __getCUDAError(r);
		default:
			abort();
	}
}

#ifdef __cplusplus
}
#endif


// CUDA Texture related functions

struct __textureReference {
	int normalized;
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureAddressMode addressMode[3];
	struct cudaChannelFormatDesc channelDesc;
	CUtexref __texref;
	int __reserved[16 - sizeof(CUtexref)];
};

std::list<CUtexref *> __textures;

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

void __cudaRegisterTexture(void **fatCubinHandle, const struct textureReference *hostVar,
		const void **deviceAddress, const char *deviceName, int dim, int norm, int ext)
{
	CUmodule *mod = (CUmodule *)fatCubinHandle;
	assert(mod != NULL);
	struct __textureReference *ref = (struct __textureReference *)hostVar;
	assert(ref != NULL);
	CUresult r = cuModuleGetTexRef(&ref->__texref, *mod, deviceName);
	assert(r == CUDA_SUCCESS);
}


cudaError_t cudaBindTextureToArray(const struct textureReference *texref,
		const struct cudaArray *array, const struct cudaChannelFormatDesc *desc)
{
	struct __textureReference *ref = (struct __textureReference *)texref;
	CUresult r;
	for(int i = 0; i < 3; i++) {
		r = cuTexRefSetAddressMode(ref->__texref, i, __getAddressMode(ref->addressMode[i]));
		if(r != CUDA_SUCCESS) return __getCUDAError(r);
	}
	r = cuTexRefSetFlags(ref->__texref, CU_TRSF_READ_AS_INTEGER);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	r = cuTexRefSetFilterMode(ref->__texref, __getFilterMode(ref->filterMode));
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	TRACE("cudaBindTextureToArray %d %d", __getChannelFormatKind(&ref->channelDesc), __getNumberOfChannels(&ref->channelDesc));
	r = cuTexRefSetFormat(ref->__texref, __getChannelFormatKind(&ref->channelDesc),
			__getNumberOfChannels(&ref->channelDesc));
	if(r != CUDA_SUCCESS) return __getCUDAError(r);
	r = cuTexRefSetArray(ref->__texref, (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);
	if(r != CUDA_SUCCESS) return __getCUDAError(r);

	__textures.push_back(&ref->__texref);

	return __getCUDAError(r);
}

cudaError_t cudaUnbindTexture(const struct textureReference *texref)
{
	struct __textureReference *ref = (struct __textureReference *)texref;
	__textures.remove(&ref->__texref);
	CUresult r = cuTexRefDestroy(ref->__texref);
	return __getCUDAError(r);
}

#ifdef __cplusplus
}
#endif

