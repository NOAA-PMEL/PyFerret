# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import numpy
import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> nogrid = pyferret.FerGrid()"
nogrid = pyferret.FerGrid()

print ">>> print repr(nogrid)"
print repr(nogrid)

print ">>> coads = pyferret.FerDSet('coads_climatology')"
coads = pyferret.FerDSet('coads_climatology')
print ">>> coads.sst.load()"
coads.sst.load()
print ">>> coadsgrid = coads.sst._datagrid"
coadsgrid = coads.sst._datagrid
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
print ">>> dupgrid._axes[0] is coadsgrid._axes[0]"
dupgrid._axes[0] is coadsgrid._axes[0]
print ">>> dupgrid._axes[0] == coadsgrid._axes[0]"
dupgrid._axes[0] == coadsgrid._axes[0]
print ">>> dupgrid._axes[1] is coadsgrid._axes[1]"
dupgrid._axes[1] is coadsgrid._axes[1]
print ">>> dupgrid._axes[1] == coadsgrid._axes[1]"
dupgrid._axes[1] == coadsgrid._axes[1]
print ">>> dupgrid._axes[2] is coadsgrid._axes[2]"
dupgrid._axes[2] is coadsgrid._axes[2]
print ">>> dupgrid._axes[2] == coadsgrid._axes[2]"
dupgrid._axes[2] == coadsgrid._axes[2]
print ">>> dupgrid._axes[3] is coadsgrid._axes[3]"
dupgrid._axes[3] is coadsgrid._axes[3]
print ">>> dupgrid._axes[3] == coadsgrid._axes[3]"
dupgrid._axes[3] == coadsgrid._axes[3]
print ">>> dupgrid._axes[4] is coadsgrid._axes[4]"
dupgrid._axes[4] is coadsgrid._axes[4]
print ">>> dupgrid._axes[4] == coadsgrid._axes[4]"
dupgrid._axes[4] == coadsgrid._axes[4]
print ">>> dupgrid._axes[5] is coadsgrid._axes[5]"
dupgrid._axes[5] is coadsgrid._axes[5]
print ">>> dupgrid._axes[5] == coadsgrid._axes[5]"
dupgrid._axes[5] == coadsgrid._axes[5]

print ">>> freqax = pyferret.FerAxis(axtype=pyferret.AXISTYPE_CUSTOM,coords=numpy.arange(1,13,0.5),unit='freqnum',name='frequencies')"
freqax = pyferret.FerAxis(axtype=pyferret.AXISTYPE_CUSTOM,coords=numpy.arange(1,13,0.5),unit='freqnum',name='frequencies')
print ">>> print repr(freqax)"
print repr(freqax)
print ">>> freqgrid = coadsgrid.copy(ax=pyferret.T_AXIS,newax=None).copy(name='freqgrid',ax=pyferret.E_AXIS,newax=freqax)"
freqgrid = coadsgrid.copy(ax=pyferret.T_AXIS,newax=None).copy(name='freqgrid',ax=pyferret.E_AXIS,newax=freqax)
print ">>> print repr(freqgrid)"
print repr(freqgrid)
print ">>> freqgrid._axes[0] == coadsgrid._axes[0]"
freqgrid._axes[0] == coadsgrid._axes[0]
print ">>> freqgrid._axes[1] == coadsgrid._axes[1]"
freqgrid._axes[1] == coadsgrid._axes[1]
print ">>> freqgrid._axes[2] == coadsgrid._axes[2]"
freqgrid._axes[2] == coadsgrid._axes[2]
print ">>> freqgrid._axes[3] == coadsgrid._axes[3]"
freqgrid._axes[3] == coadsgrid._axes[3]
print ">>> freqgrid._axes[4] == coadsgrid._axes[4]"
freqgrid._axes[4] == coadsgrid._axes[4]
print ">>> freqgrid._axes[5] == coadsgrid._axes[5]"
freqgrid._axes[5] == coadsgrid._axes[5]

