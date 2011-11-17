#ifdef USE_DBC

#include "memory/protocol/Lazy.h"
#include "util/UniquePtr.h"

namespace __dbc { namespace memory { namespace protocol {

LazyBase::LazyBase(bool eager) :
    __impl::memory::protocol::LazyBase(eager)
{
}

LazyBase::~LazyBase()
{
}

gmacError_t
LazyBase::signal_read(BlockPtrImpl _block, hostptr_t addr)
{
	REQUIRES(_block);

    LazyBlockPtrImpl block = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::signal_read(block, addr);

    return ret;
}

gmacError_t
LazyBase::signal_write(BlockPtrImpl _block, hostptr_t addr)
{
	REQUIRES(_block);

	LazyBlockPtrImpl block = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(_block);
    gmacError_t ret = Parent::signal_write(block, addr);

    ENSURES(block->getState() == __impl::memory::protocol::lazy::Dirty);

    return ret;
}

gmacError_t
LazyBase::acquire(BlockPtrImpl _block, GmacProtection &prot)
{
	REQUIRES(_block);

	LazyBlockPtrImpl block = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(_block);

    REQUIRES(block->getState() == __impl::memory::protocol::lazy::ReadOnly ||
             block->getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::acquire(block, prot);

    ENSURES((prot != GMAC_PROT_READWRITE && prot != GMAC_PROT_WRITE) ||
            block->getState() == __impl::memory::protocol::lazy::Invalid);

    return ret;
}

gmacError_t
LazyBase::release(BlockPtrImpl _block)
{
	REQUIRES(_block);
	LazyBlockPtrImpl block = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(_block);
    gmacError_t ret = Parent::release(block);

    ENSURES(block->getState() == __impl::memory::protocol::lazy::ReadOnly ||
            block->getState() == __impl::memory::protocol::lazy::Invalid);

    return ret;
}

gmacError_t
LazyBase::releaseAll()
{
    gmacError_t ret = Parent::releaseAll();

    ENSURES(Parent::dbl_.size() == 0);

    return ret;
}

gmacError_t
LazyBase::toHost(BlockPtrImpl _block)
{
	REQUIRES(_block);
	LazyBlockPtrImpl block = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(_block);
    gmacError_t ret = Parent::toHost(block);

    ENSURES(block->getState() != __impl::memory::protocol::lazy::Invalid);

    return ret;
}

__impl::hal::event_t
LazyBase::memset(const BlockPtrImpl block, size_t blockOffset, int v, size_t size, gmacError_t &err)
{
	REQUIRES(block);
    REQUIRES(blockOffset + size <= block->size());

    __impl::hal::event_t ret = Parent::memset(block, blockOffset, v, size, err);

    return ret;
}

gmacError_t
LazyBase::flushDirty()
{
    gmacError_t ret = Parent::flushDirty();

    ENSURES(Parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_t
LazyBase::copyBlockToBlock(BlockPtrImpl d, size_t dstOffset, BlockPtrImpl s, size_t srcOffset, size_t count, gmacError_t &err)
{
	REQUIRES(d);
	REQUIRES(s);

    LazyBlockPtrImpl dst = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(d);
    LazyBlockPtrImpl src = __impl::util::smart_ptr<LazyBlockImpl>::static_pointer_cast(s);

    REQUIRES(dstOffset + count <= dst->size());
    REQUIRES(srcOffset + count <= src->size());

    StateImpl dstState = dst->getState();
    StateImpl srcState = src->getState();

    __impl::hal::event_t ret = Parent::copyBlockToBlock(d, dstOffset, s, srcOffset, count, err);

    ENSURES(dst->getState() == dstState);
    ENSURES(src->getState() == srcState);

    return ret;
}

}}}

#endif // USE_DBC
