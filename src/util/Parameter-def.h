// GMAC global settings
PARAM(paramProtocol, const char *, "Rolling", "GMAC_PROTOCOL")
PARAM(paramAllocator, const char *, "Slab", "GMAC_ALLOCATOR")
PARAM(paramAcquireOnWrite, bool, false, "GMAC_ACQUIRE_ON_WRITE")
PARAM(paramIOMemory, size_t, 16 * 1024 * 1024, "GMAC_IOMEMORY")

// GMAC debug settings
PARAM(paramDebug, const char *, "none", "GMAC_DEBUG")
//PARAM(paramDebugFile, const char *, NULL, "GMAC_DEBUG_FILE")

// GMAC Page table settings
PARAM(paramPageSize, size_t, 2 * 1024 * 1024, "GMAC_PAGE", PARAM_NONZERO)

// GMAC Bitmap settings
PARAM(paramBitmapChunksPerPage, size_t, 1, "GMAC_BITMAP_CHUNKS", PARAM_NONZERO)

// Rolling Manager specific settings
PARAM(paramRollSize, size_t, 2, "GMAC_ROLL_SIZE", PARAM_NONZERO)

// Context specific settings
PARAM(paramBufferPageLockedSize, size_t, 1, "GMAC_BUFFER_PAGE_LOCKED_SIZE", PARAM_NONZERO)

// Miscelaneous parameters
PARAM(configPrintParams, bool, false, "GMAC_PRINT_PARAMS")

// OpenCL parameters
PARAM(paramOpenCLSources, const char *, "", "GMAC_OPENCL_SOURCES")
PARAM(paramOpenCLFlags,   const char *, "", "GMAC_OPENCL_FLAGS")

// GMAC parameters for auto-tunning
PARAM(paramModelToHostConfig, float, 40.0, "GMAC_MODEL_TOHOSTCONFIG")       // DMA configuration costs
PARAM(paramModelToHostTransferL1, float, 0.0007f, "GMAC_MODEL_TOHOSTTRANSFER_L1") // Transfer costs for data that fits in the L1 cache
PARAM(paramModelToHostTransferL2, float, 0.0008f, "GMAC_MODEL_TOHOSTTRANSFER_L2") // Transfer costs for data that fits in the L2 cache
PARAM(paramModelToHostTransferMem, float, 0.0010f, "GMAC_MODEL_TOHOSTTRANSFER_L2") // Transfer costs for data that does not fit in the L2 cache
PARAM(paramModelToDeviceConfig, float, 40.0, "GMAC_MODEL_TODEVICECONFIG")       // DMA configuration costs
PARAM(paramModelToDeviceTransferL1, float, 0.0007f, "GMAC_MODEL_TODEVICETRANSFER_L1") // Transfer costs for data that fits in the L1 cache
PARAM(paramModelToDeviceTransferL2, float, 0.0008f, "GMAC_MODEL_TODEVICETRANSFER_L2") // Transfer costs for data that fits in the L2 cache
PARAM(paramModelToDeviceTransferMem, float, 0.0010f, "GMAC_MODEL_TODEVICETRANSFER_L2") // Transfer costs for data that does not fit in the L2 cache
PARAM(paramModelL1, size_t, 32 * 1024, "GMAC_MODEL_L1") // Size of the L1 cache
PARAM(paramModelL2, size_t, 256 * 1024, "GMAC_MODEL_L2") // Size of the L2 cache
