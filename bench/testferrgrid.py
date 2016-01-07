# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import numpy
import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> nogrid = pyferret.FerrGrid()"
nogrid = pyferret.FerrGrid()

print ">>> print repr(nogrid)"
print repr(nogrid)

print ">>> coads = pyferret.FerrDataSet('coads_climatology')"
coads = pyferret.FerrDataSet('coads_climatology')
print ">>> coads.sst.fetch()"
coads.sst.fetch()
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
print ">>> dupgrid._axiscoords[0] is coadsgrid._axiscoords[0]"
dupgrid._axiscoords[0] is coadsgrid._axiscoords[0]
print ">>> numpy.allclose(dupgrid._axiscoords[0], coadsgrid._axiscoords[0])"
numpy.allclose(dupgrid._axiscoords[0], coadsgrid._axiscoords[0])
print ">>> dupgrid._axiscoords[1] is coadsgrid._axiscoords[1]"
dupgrid._axiscoords[1] is coadsgrid._axiscoords[1]
print ">>> numpy.allclose(dupgrid._axiscoords[1], coadsgrid._axiscoords[1])"
numpy.allclose(dupgrid._axiscoords[1], coadsgrid._axiscoords[1])
print ">>> print repr(dupgrid._axiscoords[2])"
print repr(dupgrid._axiscoords[2])
print ">>> dupgrid._axiscoords[3] is coadsgrid._axiscoords[3]"
dupgrid._axiscoords[3] is coadsgrid._axiscoords[3]
print ">>> numpy.allclose(dupgrid._axiscoords[3], coadsgrid._axiscoords[3])"
numpy.allclose(dupgrid._axiscoords[3], coadsgrid._axiscoords[3])
print ">>> print repr(dupgrid._axiscoords[4])"
print repr(dupgrid._axiscoords[4])
print ">>> print repr(dupgrid._axiscoords[5])"
print repr(dupgrid._axiscoords[5])

print ">>> freqvals = numpy.arange(1,13,0.5)"
freqvals = numpy.arange(1,13,0.5)
print ">>> print repr(freqvals)"
print repr(freqvals)
print ">>> freqgrid = coadsgrid.copy(axis=pyferret.T_AXIS, axtype=pyferret.AXISTYPE_NORMAL).copy("
print "...     gridname='freqgrid', axis=pyferret.E_AXIS, axtype=pyferret.AXISTYPE_CUSTOM,"
print "...     axcoords=freqvals, axunit='freqnum', axname='frequencies')"
freqgrid = coadsgrid.copy(axis=pyferret.T_AXIS, axtype=pyferret.AXISTYPE_NORMAL).copy(
    gridname='freqgrid', axis=pyferret.E_AXIS, axtype=pyferret.AXISTYPE_CUSTOM, 
    axcoords=freqvals, axunit='freqnum', axname='frequencies')
print ">>> print freqgrid._gridname"
print freqgrid._gridname
print ">>> print freqgrid._axistypes"
print freqgrid._axistypes
print ">>> print freqgrid._axisnames"
print freqgrid._axisnames
print ">>> print freqgrid._axisunits"
print freqgrid._axisunits
print ">>> numpy.allclose(freqgrid._axiscoords[0], coadsgrid._axiscoords[0])"
numpy.allclose(freqgrid._axiscoords[0], coadsgrid._axiscoords[0])
print ">>> numpy.allclose(freqgrid._axiscoords[1], coadsgrid._axiscoords[1])"
numpy.allclose(freqgrid._axiscoords[1], coadsgrid._axiscoords[1])
print ">>> print repr(freqgrid._axiscoords[2])"
print repr(freqgrid._axiscoords[2])
print ">>> print repr(freqgrid._axiscoords[3])"
print repr(freqgrid._axiscoords[3])
print ">>> print repr(freqgrid._axiscoords[4])"
print repr(freqgrid._axiscoords[4])
print ">>> print repr(freqgrid._axiscoords[5])"
print repr(freqgrid._axiscoords[5])

print ">>> print repr(pyferret.FerrGrid._parsegeoval(None))"
print repr(pyferret.FerrGrid._parsegeoval(None))
print ">>> print repr(pyferret.FerrGrid._parsegeoval(0))"
print repr(pyferret.FerrGrid._parsegeoval(0))
print ">>> print repr(pyferret.FerrGrid._parsegeoval(0.0))"
print repr(pyferret.FerrGrid._parsegeoval(0.0))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('0'))"
print repr(pyferret.FerrGrid._parsegeoval('0'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('5E'))"
print repr(pyferret.FerrGrid._parsegeoval('5E'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('6W'))"
print repr(pyferret.FerrGrid._parsegeoval('6W'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('7N'))"
print repr(pyferret.FerrGrid._parsegeoval('7N'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('8S'))"
print repr(pyferret.FerrGrid._parsegeoval('8S'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('9m'))"
print repr(pyferret.FerrGrid._parsegeoval('9m'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005 06:07:08'))"
print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005 06:07:08'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005 06:07'))"
print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005 06:07'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005'))"
print repr(pyferret.FerrGrid._parsegeoval('03-APR-2005'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('2003-04-05T06:07:08'))"
print repr(pyferret.FerrGrid._parsegeoval('2003-04-05T06:07:08'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('2003-04-05T06:07'))"
print repr(pyferret.FerrGrid._parsegeoval('2003-04-05T06:07'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('2003-04-05 06:07:08'))"
print repr(pyferret.FerrGrid._parsegeoval('2003-04-05 06:07:08'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('2003-04-05 06:07'))"
print repr(pyferret.FerrGrid._parsegeoval('2003-04-05 06:07'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('2003-04-05'))"
print repr(pyferret.FerrGrid._parsegeoval('2003-04-05'))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('4y', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('4y', istimestep=True))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('6d', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('6d', istimestep=True))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('7h', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('7h', istimestep=True))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('8m', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('8m', istimestep=True))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('9s', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('9s', istimestep=True))
print ">>> print repr(pyferret.FerrGrid._parsegeoval('1', istimestep=True))"
print repr(pyferret.FerrGrid._parsegeoval('1', istimestep=True))


print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice(5,23,2) ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice(5,23,2) ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice(-5.0,15.0,4.0) ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice(-5.0,15.0,4.0) ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice('-6','11','5') ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice('-6','11','5') ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice('25W','35E',5) ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice('25W','35E',5) ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice('15S','30N',3) ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice('15S','30N',3) ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice('-900m','-100m','50m') ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice('-900m','-100m','50m') ))
print ">>> print repr(pyferret.FerrGrid._parsegeoslice( slice('03-APR-2005 11:30','23-JUL-2006 23:30','12h') ))"
print repr(pyferret.FerrGrid._parsegeoslice( slice('03-APR-2005 11:30','23-JUL-2006 23:30','12h') ))

