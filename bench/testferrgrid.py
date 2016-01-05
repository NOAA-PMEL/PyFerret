# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

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

