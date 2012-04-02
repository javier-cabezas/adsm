/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2012, Javier Cabezas <jcabezas in ac upc edu> {{{
 *
 * This program is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License 
 * as published by the Free Software Foundation; either 
 * version 2 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 * }}}
 */

#ifndef GMAC_UTIL_TAGS_H_
#define GMAC_UTIL_TAGS_H_

#include <string>

namespace __impl { namespace util {

template <typename Tag = std::string>
class GMAC_LOCAL taggeable
{
public:
    typedef std::set<Tag> set_tag;

private:
    set_tag tags_;

public:
    bool add_tag(const Tag &tag)
    {
        return tags_.insert(tag).second;
    }

    size_t add_tags(const set_tag &tags)
    {
        size_t ret = 0;

        for (auto &tag : tags) {
            if (tags_.insert(tag).second == true) {
                ++ret;
            }
        }

        return ret;
    }

    bool has_tag(const Tag &tag)
    {
        return tags_.find(tag) != tags_.end();
    }

    bool has_tags(const set_tag &tags)
    {
        for (auto &tag : tags) {
            if (tags_.find(tag) == tags_.end()) {
                return false;
            }
        }

        return true;
    }

    bool remove_tag(const Tag &tag)
    {
        return tags_.erase(tag) == 1;
    }

    size_t remove_tags(const set_tag &tags)
    {
        size_t ret = 0;

        for (auto &tag : tags) {
            ret += tags_.erase(tag);
        }

        return ret;
    }

    const static set_tag empty;
};

template <typename Tag>
const typename taggeable<Tag>::set_tag taggeable<Tag>::empty;

}}

#endif /* TAGS_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
