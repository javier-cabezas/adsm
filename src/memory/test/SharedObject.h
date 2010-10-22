#ifndef GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_
#define GMAC_MEMORY_TEST_SHAREDOBJECT_IPP_

#include "memory/SharedObject.h"

namespace gmac { namespace memory {

template<typename T>
class GMAC_LOCAL SharedObjectTest :
    public SharedObjectImpl<T>,
    public virtual gmac::test::Contract {
public:
    SharedObjectTest(size_t size, T init);
    virtual ~SharedObjectTest();

    void init();
    void fini();

    // To host functions
    gmacError_t toHost(Block &block) const;
    gmacError_t toHost(Block &block, unsigned blockOff, size_t count) const;
    gmacError_t toHostPointer(Block &block, unsigned blockOff, void *ptr, size_t count) const;
    gmacError_t toHostBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    // To accelerator functions
    gmacError_t toAccelerator(Block &block) const;
    gmacError_t toAccelerator(Block &block, unsigned blockOff, size_t count) const;
    gmacError_t toAcceleratorFromPointer(Block &block, unsigned blockOff, const void *ptr, size_t count) const;
    gmacError_t toAcceleratorFromBuffer(Block &block, unsigned blockOff, IOBuffer &buffer, unsigned bufferOff, size_t count) const;

    Mode &owner() const;
    gmacError_t realloc(Mode &mode);
};

}}

#include "SharedObject.ipp"

#endif
