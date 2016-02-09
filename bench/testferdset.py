# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads = pyferret.FerDSet('coads_climatology.cdf')"
coads = pyferret.FerDSet('coads_climatology.cdf')
print ">>> str(coads)"
str(coads)
print ">>> dir(coads)"
dir(coads)
print ">>> coads.show()"
coads.show()

print ">>> coads_uw  = pyferret.FerDSet('coads_uw.nc')"
coads_uw  = pyferret.FerDSet('coads_uw.nc')
print ">>> str(coads_uw)"
str(coads_uw)
print ">>> dir(coads_uw)"
dir(coads_uw)
print ">>> coads_uw.show(brief=False)"
coads_uw.show(brief=False)

print ">>> anond = pyferret.FerDSet(None)"
anond = pyferret.FerDSet(None)
print ">>> print repr(anond)"
print repr(anond)
print ">>> dir(anond)"
dir(anond)
print ">>> anond.show()"
anond.show()
print ">>> anond.sstcopy = coads.sst"
anond.sstcopy = coads.sst
print ">>> anond.show()"
anond.show()

print ">>> pyferret.showdata(brief=False)"
pyferret.showdata(brief=False)
print ">>> anond.show()"
anond.show()
print ">>> coads_uw.close()"
coads_uw.close()
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> del coads_uw"
del coads_uw
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> coads.close()"
coads.close()
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> coads.close()"
coads.close()
print ">>> pyferret.showdata()"
pyferret.showdata()
print ">>> del coads"
del coads
print ">>> pyferret.showdata()"
pyferret.showdata()

print ">>> dir(anond)"
dir(anond)
print ">>> anond.show()"
anond.show()
print ">>> anond.close()"
anond.close()
print ">>> dir(anond)"
dir(anond)
print ">>> anond.show()"
anond.show()
print ">>> anond.close()"
anond.close()
print ">>> del anond"
del anond

