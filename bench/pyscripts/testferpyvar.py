# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

from __future__ import print_function

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print()

print(">>> coads = pyferret.FerDSet('coads_climatology.cdf')")
coads = pyferret.FerDSet('coads_climatology.cdf')
print(">>> pyferret.showdata(brief=False)")
pyferret.showdata(brief=False)

print(">>> coads.sst.load()")
coads.sst.load()
print(">>> datacopy = coads.sst.data")
datacopy = coads.sst.data
print(">>> datacopy[87:93,43:47,0,:,0,0] = -5.0")
datacopy[87:93,43:47,0,:,0,0] = -5.0
print(">>> coads.sstcopy = pyferret.FerPyVar(grid=coads.sst.grid, data=datacopy, missval=coads.sst.missval, unit=coads.sst.unit, title='modified copy of SST')")
coads.sstcopy = pyferret.FerPyVar(grid=coads.sst.grid, data=datacopy, missval=coads.sst.missval, unit=coads.sst.unit, title='modified copy of SST')
print(">>> pyferret.showdata(brief=False)")
pyferret.showdata(brief=False)
print(">>> pyferret.setwindow(1, axisasp=0.5, logo=False)")
pyferret.setwindow(1,axisasp=0.5,logo=False)
print(">>> pyferret.setdefaulttext(font='lucida')")
pyferret.settextstyle(font='lucida')
print(">>> pyferret.shadeplot(coads.sstcopy['40S':'40N','100E':'100W','16-FEB'])")
pyferret.shadeplot(coads.sstcopy['40S':'40N','100E':'100W','16-FEB'])
print(">>> pyferret.saveplot('testferpyvar.pdf')")
pyferret.saveplot('testferpyvar.pdf')

# print ">>> pyferret.run('let sstdiff = sst - sstcopy')"
# pyferret.run('let sstdiff = sst - sstcopy')
# print ">>> pyferret.run('vtree sstdiff')"
# pyferret.run('vtree sstdiff')

print(">>> del coads.sstcopy")
del coads.sstcopy
print(">>> pyferret.showdata()")
pyferret.showdata()
print(">>> coads.close()")
coads.close()
print(">>> pyferret.showdata()")
pyferret.showdata()

