#!/bin/env python

from os import getenv, system
import sys
# generate dummy rcooc file for pwz initialize. word id is from No 1, not zero.
# after run this script, you should split it via linesplit.py

totalwords = int(sys.argv[1])
for i in xrange(totalwords):
    print i+1, " 1 ", " 8 1"






