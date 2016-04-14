# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import numpy
import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> nogrid = pyferret.FerGrid()"
nogrid = pyferret.FerGrid()
print ">>> print repr(nogrid)"
print repr(nogrid)
print ">>> dir(nogrid)"
dir(nogrid)

print ">>> coads = pyferret.FerDSet('coads_climatology')"
coads = pyferret.FerDSet('coads_climatology')
print ">>> coads.sst.load()"
coads.sst.load()
print ">>> coadsgrid = coads.sst.grid"
coadsgrid = coads.sst.grid
print ">>> print repr(coadsgrid)"
print repr(coadsgrid)
print ">>> nogrid == coadsgrid"
nogrid == coadsgrid
print ">>> nogrid != coadsgrid"
nogrid != coadsgrid

print ">>> dupgrid = coadsgrid.copy()"
dupgrid = coadsgrid.copy()
print ">>> dupgrid is coadsgrid"
dupgrid is coadsgrid
print ">>> dupgrid == coadsgrid"
dupgrid == coadsgrid
print ">>> dupgrid.axes[0] is coadsgrid.axes[0]"
dupgrid.axes[0] is coadsgrid.axes[0]
print ">>> dupgrid.axes[0] == coadsgrid.axes[0]"
dupgrid.axes[0] == coadsgrid.axes[0]
print ">>> dupgrid.axes[1] is coadsgrid.axes[1]"
dupgrid.axes[1] is coadsgrid.axes[1]
print ">>> dupgrid.axes[1] == coadsgrid.axes[1]"
dupgrid.axes[1] == coadsgrid.axes[1]
print ">>> dupgrid.axes[2] is coadsgrid.axes[2]"
dupgrid.axes[2] is coadsgrid.axes[2]
print ">>> dupgrid.axes[2] == coadsgrid.axes[2]"
dupgrid.axes[2] == coadsgrid.axes[2]
print ">>> dupgrid.axes[3] is coadsgrid.axes[3]"
dupgrid.axes[0] is coadsgrid.axes[0]
print ">>> dupgrid.axes[3] == coadsgrid.axes[3]"
dupgrid.axes[3] == coadsgrid.axes[3]
print ">>> dupgrid.axes[4] is coadsgrid.axes[4]"
dupgrid.axes[4] is coadsgrid.axes[4]
print ">>> dupgrid.axes[4] == coadsgrid.axes[4]"
dupgrid.axes[4] == coadsgrid.axes[4]
print ">>> dupgrid.axes[5] is coadsgrid.axes[5]"
dupgrid.axes[0] is coadsgrid.axes[0]
print ">>> dupgrid.axes[5] == coadsgrid.axes[5]"
dupgrid.axes[5] == coadsgrid.axes[5]

print ">>> freqax = pyferret.FerAxis(axtype=pyferret.AXISTYPE_CUSTOM,coords=numpy.arange(1,13,0.5),unit='freqnum',name='frequencies')"
freqax = pyferret.FerAxis(axtype=pyferret.AXISTYPE_CUSTOM,coords=numpy.arange(1,13,0.5),unit='freqnum',name='frequencies')
print ">>> print repr(freqax)"
print repr(freqax)
print ">>> freqgrid = coadsgrid.copy(ax=pyferret.T_AXIS,newax=None).copy(name='freqgrid',ax=pyferret.E_AXIS,newax=freqax)"
freqgrid = coadsgrid.copy(ax=pyferret.T_AXIS,newax=None).copy(name='freqgrid',ax=pyferret.E_AXIS,newax=freqax)
print ">>> print repr(freqgrid)"
print repr(freqgrid)
print ">>> freqgrid.axes[0] == coadsgrid.axes[0]"
freqgrid.axes[0] == coadsgrid.axes[0]
print ">>> freqgrid.axes[1] == coadsgrid.axes[1]"
freqgrid.axes[1] == coadsgrid.axes[1]
print ">>> freqgrid.axes[2] == coadsgrid.axes[2]"
freqgrid.axes[2] == coadsgrid.axes[2]
print ">>> freqgrid.axes[3] == coadsgrid.axes[3]"
freqgrid.axes[3] == coadsgrid.axes[3]
print ">>> freqgrid.axes[4] == coadsgrid.axes[4]"
freqgrid.axes[4] == coadsgrid.axes[4]
print ">>> freqgrid.axes[5] == coadsgrid.axes[5]"
freqgrid.axes[5] == coadsgrid.axes[5]

