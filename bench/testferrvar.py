# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads = pyferret.FerrDataSet('coads_climatology.cdf')"
coads = pyferret.FerrDataSet('coads_climatology.cdf')
print ">>> str(coads.sst)"
print str(coads.sst)
print ">>> coads.sst.showgrid()"
print coads.sst.showgrid()
print ">>> del coads.sst"
del coads.sst

print ">>> coads_uw  = pyferret.FerrDataSet('coads_uw.nc')"
coads_uw  = pyferret.FerrDataSet('coads_uw.nc')
print ">>> coads_uw.uwnd.showgrid()"
coads_uw.uwnd.showgrid()

print ">>> coads_uw.mywnd = (coads_uw.uwnd**2 + coads_uw.vwnd**2)**0.5"
coads_uw.mywnd = (coads_uw.uwnd**2 + coads_uw.vwnd**2)**0.5
print ">>> dir(coads_uw)"
dir(coads_uw)
print ">>> print repr(coads_uw.mywnd)"
print repr(coads_uw.mywnd)
print ">>> coads_uw.mywnd.showgrid()"
coads_uw.mywnd.showgrid()

print ">>> print repr(coads_uw.mywnd._dataarray)"
print repr(coads_uw.mywnd._dataarray)
print ">>> print repr(coads_uw.mywnd._datagrid)"
print repr(coads_uw.mywnd._datagrid)
print ">>> coads_uw.mywnd.fetch()"
coads_uw.mywnd.fetch()
print ">>> coads_uw.mywnd._dataarray.shape"
coads_uw.mywnd._dataarray.shape
print ">>> print repr(coads_uw.mywnd._dataarray[2:5,2:5,0,0,0,0])"
print repr(coads_uw.mywnd._dataarray[2:5,2:5,0,0,0,0])
print ">>> print repr(coads_uw.mywnd._datagrid)"
print repr(coads_uw.mywnd._datagrid)
print ">>> coads_uw.mywnd.clean()"
coads_uw.mywnd.clean()

print ">>> coads.sstregrid = coads.sst.regrid(coads_uw.mywnd, pyferret.REGRID_AVERAGE)"
coads.sstregrid = coads.sst.regrid(coads_uw.mywnd, pyferret.REGRID_AVERAGE)
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> coads.sstregrid.showgrid()"
coads.sstregrid.showgrid()
print ">>> coads.sstregrid.fetch()"
coads.sstregrid.fetch()
print ">>> coads.sstregrid._dataarray.shape"
coads.sstregrid._dataarray.shape
print ">>> print repr(coads.sstregrid._dataarray[2:5,2:5,0,0,0,0])"
print repr(coads.sstregrid._dataarray[2:5,2:5,0,0,0,0])
print "pyferret.run('LIST /X=55W:51W /Y=5N:9N /L=1 SST[d=coads_climatology]')"
pyferret.run('LIST /X=55W:51W /Y=5N:9N /L=1 SST[d=coads_climatology]')

print ">>> coads_uw.showdataset()"
coads_uw.showdataset()
print ">>> del coads_uw.mywnd"
del coads_uw.mywnd
print ">>> coads_uw.showdataset()"
coads_uw.showdataset()

print ">>> anond = pyferret.FerrDataSet('')"
anond = pyferret.FerrDataSet('')
print ">>> anond.sstcopy = coads.sst"
anond.sstcopy = coads.sst
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> print repr(anond.sstcopy)"
print repr(anond.sstcopy)
print ">>> print repr(anond.sstcopy._isfilevar)"
print repr(anond.sstcopy._isfilevar)
print ">>> print repr(anond.sstcopy._requires)"
print repr(anond.sstcopy._requires)
print ">>> print repr(coads.sst)"
print repr(coads.sst)
print ">>> print repr(coads.sst._isfilevar)"
print repr(coads.sst._isfilevar)
print ">>> print repr(coads.sst._requires)"
print repr(coads.sst._requires)
print ">>> anond.close()"
anond.close()
print ">>> anond.showdataset()"
anond.showdataset()
print ">>> del anond"
del anond

print ">>> pyferret.showdatasets(brief=False)"
pyferret.showdatasets(brief=False)
print ">>> coads_uw.close()"
coads_uw.close()
print ">>> del coads_uw"
del coads_uw
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> coads.close()"
coads.close()
print ">>> del coads"
del coads
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()

