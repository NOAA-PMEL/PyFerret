# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads = pyferret.FerDSet('coads_climatology.cdf')"
coads = pyferret.FerDSet('coads_climatology.cdf')
print ">>> str(coads.sst)"
print str(coads.sst)
print ">>> coads.sst.showgrid()"
print coads.sst.showgrid()
print ">>> del coads.sst"
del coads.sst

print ">>> coads_uw  = pyferret.FerDSet('coads_uw.nc')"
coads_uw  = pyferret.FerDSet('coads_uw.nc')
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

print ">>> print repr(coads_uw.mywnd.data)"
print repr(coads_uw.mywnd.data)
print ">>> print repr(coads_uw.mywnd.grid)"
print repr(coads_uw.mywnd.grid)
print ">>> coads_uw.mywnd.load()"
coads_uw.mywnd.load()
print ">>> coads_uw.mywnd.data.shape"
coads_uw.mywnd.data.shape
print ">>> print repr(coads_uw.mywnd.data[2:5,2:5,0,0,0,0])"
print repr(coads_uw.mywnd.data[2:5,2:5,0,0,0,0])
print ">>> print repr(coads_uw.mywnd.grid)"
print repr(coads_uw.mywnd.grid)
print ">>> coads_uw.mywnd.unload()"
coads_uw.mywnd.unload()

print ">>> coads.sstregrid = coads.sst.regrid(coads_uw.mywnd, pyferret.REGRID_AVERAGE)"
coads.sstregrid = coads.sst.regrid(coads_uw.mywnd, pyferret.REGRID_AVERAGE)
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> coads.sstregrid.showgrid()"
coads.sstregrid.showgrid()
print ">>> coads.sstregrid.load()"
coads.sstregrid.load()
print ">>> coads.sstregrid.data.shape"
coads.sstregrid.data.shape
print ">>> print repr(coads.sstregrid.data[2:5,2:5,0,0,0,0])"
print repr(coads.sstregrid.data[2:5,2:5,0,0,0,0])
print "pyferret.run('LIST /X=55W:51W /Y=5N:9N /L=1 SST[d=coads_climatology]')"
pyferret.run('LIST /X=55W:51W /Y=5N:9N /L=1 SST[d=coads_climatology]')

print ">>> coads_uw.show()"
coads_uw.show()
print ">>> del coads_uw.mywnd"
del coads_uw.mywnd
print ">>> coads_uw.show()"
coads_uw.show()

print ">>> pyferret.anondset.sstcopy = coads.sst"
pyferret.anondset.sstcopy = coads.sst
print ">>> pyferret.anondset.show()"
pyferret.anondset.show()
print ">>> print repr(pyferret.anondset.sstcopy)"
print repr(pyferret.anondset.sstcopy)
print ">>> print repr(pyferret.anondset.sstcopy._isfilevar)"
print repr(pyferret.anondset.sstcopy._isfilevar)
print ">>> print repr(pyferret.anondset.sstcopy._requires)"
print repr(pyferret.anondset.sstcopy._requires)
print ">>> print repr(coads.sst)"
print repr(coads.sst)
print ">>> print repr(coads.sst._isfilevar)"
print repr(coads.sst._isfilevar)
print ">>> print repr(coads.sst._requires)"
print repr(coads.sst._requires)
print ">>> pyferret.anondset.close()"
pyferret.anondset.close()
print ">>> pyferret.anondset.show()"
pyferret.anondset.show()

print ">>> coads.sst2 = coads.sst['15-FEB']"
coads.sst2 = coads.sst['15-FEB']
print ">>> coads.sst2.showgrid()"
coads.sst2.showgrid()
print ">>> coads.sst2.load()"
coads.sst2.load()
print ">>> coads.sst2.data.shape"
coads.sst2.data.shape
print ">>> print repr(coads.sst2.data[2:5,23:26,0,0,0,0])"
print repr(coads.sst2.data[2:5,23:26,0,0,0,0])
print "pyferret.run('LIST /X=25E:29E /Y=43S:39S /L=2 SST[d=coads_climatology]')"
pyferret.run('LIST /X=25E:29E /Y=43S:39S /L=2 SST[d=coads_climatology]')

print ">>> coads.sst2 = coads.sst['43S':'39S','25E':'29E',:,1]"
coads.sst2 = coads.sst['43S':'39S','25E':'29E',:,1]
print ">>> print repr(coads.sst2)"
print repr(coads.sst2)
print ">>> coads.show(brief=False)"
coads.show(brief=False)
print ">>> coads.sst2.load()"
coads.sst2.load()
print ">>> coads.sst2.data.shape"
coads.sst2.data.shape
print ">>> print repr(coads.sst2.data[:,:,0,0,0,0])"
print repr(coads.sst2.data[:,:,0,0,0,0])


print ">>> pyferret.showdata(brief=False)"
pyferret.showdata(brief=False)
print ">>> coads_uw.close()"
coads_uw.close()
print ">>> del coads_uw"
del coads_uw
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> coads.close()"
coads.close()
print ">>> del coads"
del coads
print ">>> pyferret.showdata()"
pyferret.showdata()

