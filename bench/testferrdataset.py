# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads = pyferret.FerrDataSet('coads_climatology.cdf')"
coads = pyferret.FerrDataSet('coads_climatology.cdf')
print ">>> str(coads)"
str(coads)
print ">>> dir(coads)"
dir(coads)
print ">>> coads.showdataset()"
coads.showdataset()

print ">>> coads_uw  = pyferret.FerrDataSet('coads_uw.nc')"
coads_uw  = pyferret.FerrDataSet('coads_uw.nc')
print ">>> str(coads_uw)"
str(coads_uw)
print ">>> dir(coads_uw)"
dir(coads_uw)
print ">>> coads_uw.showdataset(brief=False)"
coads_uw.showdataset(brief=False)

print ">>> anond = pyferret.FerrDataSet(None)"
anond = pyferret.FerrDataSet(None)
print ">>> print repr(anond)"
print repr(anond)
print ">>> dir(anond)"
dir(anond)
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> anond.sstcopy = coads.sst"
anond.sstcopy = coads.sst
print ">>> anond.showdataset()"
anond.showdataset()

print ">>> pyferret.showdatasets(brief=False)"
pyferret.showdatasets(brief=False)
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> coads_uw.close()"
coads_uw.close()
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> del coads_uw"
del coads_uw
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> coads.close()"
coads.close()
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> coads.close()"
coads.close()
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> del coads"
del coads
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()

print ">>> dir(anond)"
dir(anond)
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> anond.close()"
anond.close()
print ">>> dir(anond)"
dir(anond)
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> anond.close()"
anond.close()
print ">>> del anond"
del anond

