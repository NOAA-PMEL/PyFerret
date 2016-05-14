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

print ">>> try:"
print "...     pyferret.FerDSet(None)"
print "...     print 'No error found'"
print "... except ValueError as ex:"
print "...     print 'ValueError caught with message ' + str(ex)"
try:
    pyferret.FerDSet(None)
    print 'No error found'
except ValueError as ex:
    print 'ValueError caught with message: ' + str(ex)

print ">>> pyferret.anondset.sstcopy = coads.sst"
pyferret.anondset.sstcopy = coads.sst
print ">>> pyferret.anondset.show()"
pyferret.anondset.show()

print ">>> pyferret.showdata(brief=False)"
pyferret.showdata(brief=False)
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

print ">>> dir(pyferret.anondset)"
dir(pyferret.anondset)
print ">>> pyferret.anondset.show()"
pyferret.anondset.show()
print ">>> pyferret.anondset.close()"
pyferret.anondset.close()
print ">>> dir(pyferret.anondset)"
dir(pyferret.anondset)
print ">>> pyferret.anondset.show()"
pyferret.anondset.show()

