#ifdef USE_DBC

#include "memory/protocol/Lazy.h"

namespace __dbc { namespace memory { namespace protocol {

LazyBase::LazyBase(bool eager) :
    __impl::memory::protocol::LazyBase(eager)
{
}

LazyBase::~LazyBase()
{
}

gmacError_t
LazyBase::signalRead(BlockImpl &_block, hostptr_t addr)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::signalRead(block, addr);

    return ret;
}

gmacError_t
LazyBase::signalWrite(BlockImpl &_block, hostptr_t addr)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::signalWrite(block, addr);

    ENSURES(block.getState() == __impl::memory::protocol::lazy::Dirty);

    return ret;
}

gmacError_t
LazyBase::acquire(BlockImpl &_block, GmacProtection &prot)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);

    REQUIRES(block.getState() == __impl::memory::protocol::lazy::ReadOnly ||
             block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = Parent::acquire(block, prot);

    ENSURES((prot != GMAC_PROT_READWRITE && prot != GMAC_PROT_WRITE) ||
            block.getState() == __impl::memory::protocol::lazy::Invalid);

    return ret;
}

gmacError_t
LazyBase::release(BlockImpl &_block)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::release(block);

    ENSURES(block.getState() == __impl::memory::protocol::lazy::ReadOnly ||
            block.getState() == __impl::memory::protocol::lazy::Invalid);

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
LazyBase::toHost(BlockImpl &_block)
{
    LazyBlockImpl &block = dynamic_cast<LazyBlockImpl &>(_block);
    gmacError_t ret = Parent::toHost(block);

    ENSURES(block.getState() != __impl::memory::protocol::lazy::Invalid);

    return ret;
}

__impl::hal::event_t
LazyBase::memset(const BlockImpl &block, size_t blockOffset, int v, size_t size, gmacError_t &err)
{
    REQUIRES(blockOffset + size <= block.size());

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
LazyBase::copyBlockToBlock(Block &d, size_t dstOffset, Block &s, size_t srcOffset, size_t count, gmacError_t &err)
{
    LazyBlockImpl &dst = dynamic_cast<LazyBlockImpl &>(d);
    LazyBlockImpl &src = dynamic_cast<LazyBlockImpl &>(s);

    REQUIRES(dstOffset + count <= dst.size());
    REQUIRES(srcOffset + count <= src.size());

    StateImpl dstState = dst.getState();
    StateImpl srcState = src.getState();

    __impl::hal::event_t ret = Parent::copyBlockToBlock(d, dstOffset, s, srcOffset, count, err);

    ENSURES(dst.getState() == dstState);
    ENSURES(src.getState() == srcState);

    return ret;
}

}}}

#endif // USE_DBC
