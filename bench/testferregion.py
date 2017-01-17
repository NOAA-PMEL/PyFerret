# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

from __future__ import print_function

import numpy

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print()

print(">>> emptyregion = pyferret.FerRegion()")
emptyregion = pyferret.FerRegion()
print(">>> print repr(emptyregion)")
print(repr(emptyregion))
print(">>> emptyregion._ferretqualifierstr()")
emptyregion._ferretqualifierstr()

print(">>> coordregion = pyferret.FerRegion(X=slice(-70,20),Y=slice('20N','60N'),Z=-100,T='15-DEC-2015',E=0,F='1')")
coordregion = pyferret.FerRegion(X=slice(-70,20),Y=slice('20N','60N'),Z=-100,T='15-DEC-2015',E=0,F='1')
print(">>> print repr(coordregion)")
print(repr(coordregion))
print(">>> coordregion._ferretqualifierstr()")
coordregion._ferretqualifierstr()

print(">>> indexregion = pyferret.FerRegion(I=slice(0,20),J=slice('1','60'),K=10,L='5',M=0,N='1')")
indexregion = pyferret.FerRegion(I=slice(0,20),J=slice('1','60'),K=10,L='5',M=0,N='1')
print(">>> print repr(indexregion)")
print(repr(indexregion))
print(">>> indexregion._ferretqualifierstr()")
indexregion._ferretqualifierstr()

print(">>> try:")
print("...     badregion = pyferret.FerRegion(X='70W:20E',I=slice(0,20))")
print("...     print 'Error not caught'")
print("... except ValueError as ex:")
print("...     print 'Error caught: %s' % str(ex)")
try:
    badregion = pyferret.FerRegion(X='70W:20E',I=slice(0,20))
    print('Error not caught')
except ValueError as ex:
    print('Error caught: %s' % str(ex))

