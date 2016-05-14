# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> dsetnames = ['./ens1.nc', './ens2.nc', './ens3.nc', './ens4.nc']"
dsetnames = ['./ens1.nc', './ens2.nc', './ens3.nc', './ens4.nc']
print ">>> fourfiles = pyferret.FerAggDSet(name='fourfiles', dsets=dsetnames, along='E')"
fourfiles = pyferret.FerAggDSet(name='fourfiles', dsets=dsetnames, along='E')
print ">>> fourfiles.show()"
fourfiles.show()
print ">>> print str(fourfiles)"
print str(fourfiles)
print ">>> dir(fourfiles)"
dir(fourfiles)
print ">>> fourfiles.getdsetnames()"
fourfiles.getdsetnames()
print ">>> fourfiles.getdsets()"
fourfiles.getdsets()

print ">>> fourfiles.SST.showgrid()"
fourfiles.SST.showgrid()

print ">>> fourfiles.close()"
fourfiles.close()
print ">>> print str(fourfiles)"
print str(fourfiles)
print ">>> dir(fourfiles)"
dir(fourfiles)
print ">>> fourfiles.getdsetnames()"
fourfiles.getdsetnames()
print ">>> fourfiles.getdsets()"
fourfiles.getdsets()

print ">>> del fourfiles"
del fourfiles

