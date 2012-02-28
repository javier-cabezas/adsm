#include "mapping.h"

namespace __impl { namespace dsm {

mapping::range_block
mapping::get_blocks_in_range(hal::ptr::offset_type offset, size_t count)
{
    ASSERTION(offset < size_);
    ASSERTION(offset + count <= size_);

    size_t tmp = 0;

    list_block::iterator first = blocks_.begin();
    
    while (true) {
        tmp += (*first)->get_size();
        if (tmp >= offset) break;
        first++;
    }

    list_block::iterator last = first;
    while (tmp < offset + count) {
        tmp += (*last)->get_size();
        // We increment before the check because we return an open range
        last++;
    }

    return range_block(first, last);
}

gmacError_t
mapping::dup2(mapping_ptr map1, hal::ptr::offset_type off1,
              mapping_ptr map2, hal::ptr::offset_type off2, size_t count)
{
    gmacError_t ret = gmacSuccess;

    return ret;
}

gmacError_t
mapping::dup(hal::ptr::offset_type off1, mapping_ptr map2,
             hal::ptr::offset_type off2, size_t count)
{
    gmacError_t ret = gmacSuccess;

    list_block::iterator it;
    hal::ptr::offset_type off = off2;

    // Move to the first block involved
    for (it = map2->blocks_.begin(); off < off2; off += (*it)->get_size());
    ASSERTION(off == off2, "map2 should contain a block starting @ address: %p", off2);

    // Duplicate all the blocks
    for (it = map2->blocks_.begin(); off < off2 + count; off += (*it)->get_size()) {
        ASSERTION(it != map2->blocks_.end(), "unexpected end of container");
        append(*it);
    }
    ASSERTION(off == off2 + count, "loop should end after a block boundary");

    return ret;
}

mapping::cursor_block
mapping::get_first_block(hal::ptr p)
{
    mapping::list_block::iterator it;

    size_t off = 0;
    for (it  = blocks_.begin();
         it != blocks_.end();
       ++it) {
        size_t blockSize = (*it)->get_size();

        if ((p >= addr_) &&
            (p <  addr_ + blockSize)) {
            size_t offLocal = (addr_ + blockSize).get_offset() - p.get_offset();
            cursor_block(it, offLocal, off);
            break;
        }

        off += blockSize;
    }

    return cursor_block(blocks_.end(), 0, 0);
}


mapping::list_block::iterator
mapping::split_block(list_block::iterator it, size_t offset)
{
    coherence::block_ptr blockNew = (*it)->split(offset);

    list_block::iterator ret = blocks_.insert(++it, blockNew);

    return ret;
}

mapping::cursor_block
mapping::split_block(cursor_block cursor, size_t offset)
{
    coherence::block_ptr blockNew = cursor.get_block()->split(offset);

    list_block::iterator it = blocks_.insert(++cursor.get_iterator(), blockNew);

    return cursor_block(it, cursor.get_offset_block(), cursor.get_offset_local());
}

gmacError_t
mapping::prepend(coherence::block_ptr b)
{
    gmacError_t ret = gmacSuccess;

    if (b->get_size() <= addr_.get_offset()) {
        blocks_.push_front(b);
        addr_ -= b->get_size();
        size_ += b->get_size();
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
mapping::append(coherence::block_ptr b)
{
    gmacError_t ret = gmacSuccess;

    if (b->get_size() > 0) {
        blocks_.push_back(b);
        size_ += b->get_size();
    } else {
        ret = gmacErrorInvalidValue;
    }

    return ret;
}

gmacError_t
mapping::append(mapping_ptr map)
{
    if ((map->addr_.get_offset()  <  (addr_.get_offset() + size_)) ||
        (map->addr_.get_aspace() !=  (addr_.get_aspace()))) {
        return gmacErrorInvalidValue;
    }

    gmacError_t ret = gmacSuccess;

    // Add a new block between the mappings if needed
    if (map->get_bounds().start > get_bounds().end) {
        coherence::block_ptr b = factory_block::create(map->get_bounds().start - get_bounds().end);

        blocks_.push_back(b);

        b->register_mapping(this, size_);

        size_ += b->get_size(); 
    }

    // Insert the rest of blocks into the map
    for (list_block::iterator it  = map->blocks_.begin();
                              it != map->blocks_.end();
                              it++) {
        blocks_.push_back(*it);

        size_ += (*it)->get_size(); 
    }
      
    return ret;
}

mapping::mapping(hal::ptr addr) :
    addr_(addr),
    size_(0)
{
}

mapping::mapping(const mapping &m) :
    addr_(m.addr_),
    size_(m.size_),
    blocks_(m.blocks_)
{
}

gmacError_t
mapping::acquire(size_t offset, size_t count, int flags)
{
    gmacError_t err = gmacSuccess;

    range_block range = get_blocks_in_range(offset, count);

    for (range_block::iterator i = range.begin(); i != range.end(); ++i) {
        err = (*i)->acquire(this, flags);
        if (err != gmacSuccess) break;
    }

    return err;
}

gmacError_t
mapping::release(size_t offset, size_t count)
{
    gmacError_t err = gmacSuccess;

    range_block range = get_blocks_in_range(offset, count);

    for (range_block::iterator i = range.begin(); i != range.end(); i++) {
        err = (*i)->release(this);
        if (err != gmacSuccess) break;
    }

    return err;
}

gmacError_t
mapping::link(hal::ptr ptrDst, mapping_ptr mDst,
              hal::ptr ptrSrc, mapping_ptr mSrc, size_t count, int flags)
{
    ASSERTION(bool(ptrDst));
    ASSERTION(bool(ptrSrc));

    ASSERTION(mDst != mSrc);

    ASSERTION(count > 0);

    ASSERTION(long_t(ptrDst.get_offset()) % MinAlignment == 0);
    ASSERTION(long_t(ptrSrc.get_offset()) % MinAlignment == 0);

    ASSERTION(ptrDst > mDst->get_ptr());
    ASSERTION(ptrSrc > mSrc->get_ptr());

    gmacError_t ret = gmacSuccess;

    cursor_block cursorDst = mDst->get_first_block(ptrDst);
    cursor_block cursorSrc = mSrc->get_first_block(ptrSrc);

    // If we are mapping in the middle of a block...
    if (cursorDst.get_offset_local() > 0) {
        // ... split the block and move to the newly created block
        mDst->split_block(cursorDst, cursorDst.get_offset_local());
        cursorDst.advance_block();
    }
    // If we are mapping in the middle of a block...
    if (cursorSrc.get_offset_local() > 0) {
        // ... split the block and move to the newly created block
        mSrc->split_block(cursorSrc, cursorSrc.get_offset_local());
        cursorSrc.advance_block();
    }

    do {
        size_t bytesSubmapping = 0;

        // Both cursors MUST point at the beginning of a block
        ASSERTION(cursorDst.get_offset_local() == 0);
        ASSERTION(cursorSrc.get_offset_local() == 0);

        // Register the mappings in the blocks
        cursorDst.get_block()->register_mapping(mSrc, cursorSrc.get_offset_block());
        cursorSrc.get_block()->register_mapping(mDst, cursorDst.get_offset_block());

        if ((cursorDst.get_block()->get_size() >= count) &&
            (cursorSrc.get_block()->get_size() >= count)) {
            // If the blocks are bigger than the mapping, split them
            if (cursorDst.get_block()->get_size() > count) {
                mDst->split_block(cursorDst, count);
            }
            if (cursorSrc.get_block()->get_size() > count) {
                mSrc->split_block(cursorSrc, count);
            }

            bytesSubmapping = count;
        } else {
            if ((cursorDst.get_block()->get_size() < count) &&
                (cursorSrc.get_block()->get_size() < count)) {
                // If both blocks are smaller than the remaining size of the mapping
                if (cursorDst.get_block()->get_size() ==
                    cursorSrc.get_block()->get_size()) {
                    // If both blocks are equally sized, advance both cursors
                    cursorDst.advance_block();
                    cursorSrc.advance_block();

                    bytesSubmapping = cursorDst.get_block()->get_size();
                } else if (cursorDst.get_block()->get_size() <
                           cursorSrc.get_block()->get_size()) {
                    // Move to next block in dst
                    size_t remainder = cursorDst.advance_block();
                    // Split src and move to the newly created block
                    mSrc->split_block(cursorSrc, remainder);
                    cursorSrc.advance_block();

                    bytesSubmapping = remainder;
                } else if (cursorSrc.get_block()->get_size() <
                           cursorDst.get_block()->get_size()) {
                    // Move to next block in src
                    size_t remainder = cursorSrc.advance_block();
                    // Split dst and move to the newly created block
                    mDst->split_block(cursorDst, remainder);
                    cursorDst.advance_block();

                    bytesSubmapping = remainder;
                }
            } else if (cursorDst.get_block()->get_size() < count) {
                // If dst block is smaller than the remaining size of the mapping

                // Move to next block in dst
                size_t remainder = cursorDst.advance_block();
                // Split src and move to the newly created block
                mSrc->split_block(cursorSrc, remainder);
                cursorSrc.advance_block();

                bytesSubmapping = remainder;
            } else /*  cursorSrc.get_block()->get_size() < count */ {
                // If src block is smaller than the remaining size of the mapping

                // Move to next block in src
                size_t remainder = cursorSrc.advance_block();
                // Split dst and move to the newly created block
                mDst->split_block(cursorDst, remainder);
                cursorDst.advance_block();

                bytesSubmapping = remainder;
            }
        }

        count -= bytesSubmapping;
    } while (count > 0);

    return ret;
}



}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
