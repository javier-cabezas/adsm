#!/usr/bin/env python
# --------------------------------------------
# <+ PROGRAM_NAME +> - <+ SHORT_DESC +>
# --------------------------------------------
# File:   ../scripts/parse.sh
# Author: Javier Cabezas <jcabezas in ac upc edu>
#
# Copyright (C) 2012, Javier Cabezas <jcabezas in ac upc edu> {{{
#
# This program is free software; you can redistribute it and/or 
# modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation; either 
# version 2 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
# }}}
#
# <+ DESCRIPTION +>

import re
import sys

BEGIN_COLOR = "\033["
END_COLOR   = "\033[1;m"

NORMAL = "0;"
BOLD   = "1;"

RED    = "31m"
GREEN  = "32m"
YELLOW = "33m"
BLUE   = "34m"
PURPLE = "35m"
CYAN   = "36m"
WHITE  = "37m"

COLORS=[ RED, BLUE, GREEN, YELLOW, CYAN, PURPLE ]

PATTERN_ID=re.compile('<[0-9]+>')

FILTER_LINES=False

def format_match(string, color):
    iterator = PATTERN_ID.finditer(string)

    ret = ''

    for it in iterator:
        begin, end = it.span()
        
        ret += BEGIN_COLOR + BOLD   + color + string[:begin] + END_COLOR
        ret += BEGIN_COLOR + NORMAL + color + string[begin    :begin + 1] + END_COLOR
        s = string[begin + 1:].split('>')[0]
        ret += BEGIN_COLOR + BOLD   + color + s   + END_COLOR
        ret += BEGIN_COLOR + NORMAL + color + '>' + END_COLOR

    return ret

FIRST_ARG_PATTERN = 1

if len(sys.argv) > 1:
    if sys.argv[1] == "-f":
        FILTER_LINES = True
        FIRST_ARG_PATTERN += 1


PATTERNS=sys.argv[FIRST_ARG_PATTERN:]
COMPILED_PATTERNS=[]

for p in PATTERNS:
    COMPILED_PATTERNS.append(re.compile('([a-zA-Z]+::)*' + p + '<[0-9]+>'))
    #COMPILED_PATTERNS.append(re.compile(p + '<[0-9]+>'))

while True:
    line = sys.stdin.readline()[:-1]
    if not line:
        break

    index  = 0
    match  = False
    output = line
    for p in COMPILED_PATTERNS:
        matches = p.finditer(line)
        unique = []
        for f in matches:
            if f.group(0) not in unique:
                unique.append(f.group(0))

        if len(unique) > 0:
            match = True
            for m in unique:
                new = format_match(m, COLORS[index])
                output = output.replace(m, new)

        index += 1

    if not FILTER_LINES or match:
        print output

# vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab:
