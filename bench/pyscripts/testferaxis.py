# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

from __future__ import print_function

import numpy

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print()

print(">>> normax = pyferret.FerAxis()")
normax = pyferret.FerAxis()
print(">>> print repr(normax)")
print(repr(normax))
print(">>> dir(normax)")
dir(normax)

print(">>> coads = pyferret.FerDSet('coads_climatology')")
coads = pyferret.FerDSet('coads_climatology')
print(">>> coads.sst.load()")
coads.sst.load()
print(">>> sstaxes = coads.sst.grid.axes")
sstaxes = coads.sst.grid.axes
print(">>> print repr(sstaxes)")
print(repr(sstaxes))
print(">>> normax == sstaxes[0]")
normax == sstaxes[0]
print(">>> normax != sstaxes[1]")
normax != sstaxes[1]
print(">>> normax == sstaxes[2]")
normax == sstaxes[2]
print(">>> normax != sstaxes[3]")
normax != sstaxes[3]

print(">>> print repr(sstaxes[0].axtype)")
print(repr(sstaxes[0].axtype))
print(">>> print repr(sstaxes[0].coords)")
print(repr(sstaxes[0].coords))
print(">>> print repr(sstaxes[0].unit)")
print(repr(sstaxes[0].unit))
print(">>> print repr(sstaxes[0].name)")
print(repr(sstaxes[0].name))

print(">>> print repr(sstaxes[1].axtype)")
print(repr(sstaxes[1].axtype))
print(">>> print repr(sstaxes[1].coords)")
print(repr(sstaxes[1].coords))
print(">>> print repr(sstaxes[1].unit)")
print(repr(sstaxes[1].unit))
print(">>> print repr(sstaxes[1].name)")
print(repr(sstaxes[1].name))

print(">>> print repr(sstaxes[2].axtype)")
print(repr(sstaxes[2].axtype))
print(">>> print repr(sstaxes[2].coords)")
print(repr(sstaxes[2].coords))
print(">>> print repr(sstaxes[2].unit)")
print(repr(sstaxes[2].unit))
print(">>> print repr(sstaxes[2].name)")
print(repr(sstaxes[2].name))

print(">>> print repr(sstaxes[3].axtype)")
print(repr(sstaxes[3].axtype))
print(">>> print repr(sstaxes[3].coords)")
print(repr(sstaxes[3].coords))
print(">>> print repr(sstaxes[3].unit)")
print(repr(sstaxes[3].unit))
print(">>> print repr(sstaxes[3].name)")
print(repr(sstaxes[3].name))

print(">>> dupaxis = sstaxes[0].copy()")
dupaxis = sstaxes[0].copy()
print(">>> dupaxis is sstaxes[0]")
dupaxis is sstaxes[0]
print(">>> dupaxis == sstaxes[0]")
dupaxis == sstaxes[0]
print(">>> dupaxis.coords is sstaxes[0].coords")
dupaxis.coords is sstaxes[0].coords
print(">>> numpy.allclose(dupaxis.coords, sstaxes[0].coords)")
numpy.allclose(dupaxis.coords, sstaxes[0].coords)

print(">>> dupaxis = sstaxes[3].copy()")
dupaxis = sstaxes[3].copy()
print(">>> dupaxis is sstaxes[3]")
dupaxis is sstaxes[3]
print(">>> dupaxis == sstaxes[3]")
dupaxis == sstaxes[3]
print(">>> dupaxis.coords is sstaxes[3].coords")
dupaxis.coords is sstaxes[3].coords
print(">>> numpy.allclose(dupaxis.coords, sstaxes[3].coords)")
numpy.allclose(dupaxis.coords, sstaxes[3].coords)

print(">>> print repr(pyferret.FerAxis._parsegeoval(None))")
print(repr(pyferret.FerAxis._parsegeoval(None)))
print(">>> print repr(pyferret.FerAxis._parsegeoval(0))")
print(repr(pyferret.FerAxis._parsegeoval(0)))
print(">>> print repr(pyferret.FerAxis._parsegeoval(0.0))")
print(repr(pyferret.FerAxis._parsegeoval(0.0)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('0'))")
print(repr(pyferret.FerAxis._parsegeoval('0')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('5E'))")
print(repr(pyferret.FerAxis._parsegeoval('5E')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('6W'))")
print(repr(pyferret.FerAxis._parsegeoval('6W')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('7N'))")
print(repr(pyferret.FerAxis._parsegeoval('7N')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('8S'))")
print(repr(pyferret.FerAxis._parsegeoval('8S')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('9m'))")
print(repr(pyferret.FerAxis._parsegeoval('9m')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('03-APR-2005 06:07:08'))")
print(repr(pyferret.FerAxis._parsegeoval('03-APR-2005 06:07:08')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('03-APR-2005 06:07'))")
print(repr(pyferret.FerAxis._parsegeoval('03-APR-2005 06:07')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('03-APR-2005'))")
print(repr(pyferret.FerAxis._parsegeoval('03-APR-2005')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('2003-04-05T06:07:08'))")
print(repr(pyferret.FerAxis._parsegeoval('2003-04-05T06:07:08')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('2003-04-05T06:07'))")
print(repr(pyferret.FerAxis._parsegeoval('2003-04-05T06:07')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('2003-04-05 06:07:08'))")
print(repr(pyferret.FerAxis._parsegeoval('2003-04-05 06:07:08')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('2003-04-05 06:07'))")
print(repr(pyferret.FerAxis._parsegeoval('2003-04-05 06:07')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('2003-04-05'))")
print(repr(pyferret.FerAxis._parsegeoval('2003-04-05')))
print(">>> print repr(pyferret.FerAxis._parsegeoval('4y', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('4y', istimestep=True)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('6d', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('6d', istimestep=True)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('7h', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('7h', istimestep=True)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('8m', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('8m', istimestep=True)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('9s', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('9s', istimestep=True)))
print(">>> print repr(pyferret.FerAxis._parsegeoval('1', istimestep=True))")
print(repr(pyferret.FerAxis._parsegeoval('1', istimestep=True)))


print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice(5,23,2) ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice(5,23,2) )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice(-5.0,15.0,4.0) ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice(-5.0,15.0,4.0) )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice('-6','11','5') ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice('-6','11','5') )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice('25W','35E',5) ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice('25W','35E',5) )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice('15S','30N',3) ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice('15S','30N',3) )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice('-900m','-100m','50m') ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice('-900m','-100m','50m') )))
print(">>> print repr(pyferret.FerAxis._parsegeoslice( slice('03-APR-2005 11:30','23-JUL-2006 23:30','12h') ))")
print(repr(pyferret.FerAxis._parsegeoslice( slice('03-APR-2005 11:30','23-JUL-2006 23:30','12h') )))

