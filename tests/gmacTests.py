#!/usr/bin/env python
# --------------------------------------------
# <+ PROGRAM_NAME +> - <+ SHORT_DESC +>
# --------------------------------------------
# Module: ../tests
# File:   ../tests/run_tests.py
# Author: Javier Cabezas <jcabezas in ac upc edu>
#
#
# }}}
# 
# <+ DESCRIPTION +>

import os
import subprocess

class Param:
    def __init__(self, name, values):
        self.name           = name
        self.default_values = values
        self.sub_params     = {}

    def add_sub_param(self, val, sub):
        if not self.sub_params.has_key(val):
            self.sub_params[val] = [sub]
        else:
            self.sub_params[val] += [sub]


# Lazy and Rolling managers' sub parameters
page = Param("GMAC_PAGE", [2097152])

# Rolling-only sub parameters
line_size = Param("GMAC_LINESIZE", [1024])
lru_delta = Param("GMAC_LRUDELTA", [2])

# First-level parameters
manager = Param("GMAC_MANAGER", ["Batch", "Lazy", "Rolling"])
manager.add_sub_param("Rolling", page)
manager.add_sub_param("Rolling", line_size)
manager.add_sub_param("Rolling", lru_delta)

manager.add_sub_param("Lazy", page)

auto_sync   = Param("GMAC_AUTO_SYNC", [0])
page_locked = Param("GMAC_BUFFER_PAGED_LOCKED_SIZE", [4194304])

PARAMS = [manager, auto_sync, page_locked]

def recursive(params):
    if len(params) == 0:
        return []

    level = []

    base = {}
    for param in params:
        if len(param.default_values) == 1:
            base[param.name] = str(param.default_values[0])
        else:
            base[param.name] = None

    curr = [base]

    for param in params:
        if base[param.name] == None:
            curr2 = []
            for v in param.default_values:
                for d in curr:
                    d2 = {}
                    d2.update(d)
                    d2[param.name] = str(v)
                    curr2 += [d2]
            curr = curr2

    for param in params:
        for name, sub_params in param.sub_params.items():
            next_level = recursive(sub_params)
            curr2 = []
            for d in next_level:
                for d2 in curr:
                    if d2[param.name] == name:
                        d3 = {}
                        d3.update(d)
                        d3.update(d2)
                        curr2 += [d3]
                    else:
                        curr2 += [d2]

            curr = curr2

    return curr

class Test:
    TESTS=[]
    def __init__(self, name, params):
        self.name = name
        self.params = params
        Test.TESTS += [self]

    def launch_test(self):
        dicts = recursive(self.params)

        print 'Executing', self.name
        outputs = []

        for d in dicts:
            print d['GMAC_MANAGER'],
            d.update(os.environ)
            p = subprocess.Popen(['./' + self.name], env=d, stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))
            p.wait()
            output = p.stdout.readlines()[-1][:-1]
            outputs += [output]
            print output

    @staticmethod
    def launch():
        for t in Test.TESTS:
            t.launch_test()

#Test("bandwidth", PARAMS)
Test("gmacVecAdd", PARAMS)
Test("gmacThreadVecAdd", PARAMS)
Test("gmacConstant", PARAMS)
Test("gmacTexture", PARAMS)
Test("gmacMemcpy", PARAMS)
Test("gmacMemset", PARAMS)
Test("gmacMatrixMul", PARAMS)
Test("gmacPingPong", PARAMS)
Test("gmacSharedVecAdd", PARAMS)
Test("gmacCompress", PARAMS)
Test("gmacCompressSend", PARAMS)
Test("gmacStencil", PARAMS)
Test("gmacThreadStencil", PARAMS)

Test.launch()

# vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab:
