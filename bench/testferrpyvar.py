# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads = pyferret.FerrDataSet('coads_climatology.cdf')"
coads = pyferret.FerrDataSet('coads_climatology.cdf')
print ">>> pyferret.showdatasets(brief=False)"
pyferret.showdatasets(brief=False)

print ">>> coads.sst.fetch()"
coads.sst.fetch()
print ">>> datacopy = coads.sst._dataarray"
datacopy = coads.sst._dataarray
print ">>> datacopy[87:93,43:47,0,:,0,0] = -5.0"
datacopy[87:93,43:47,0,:,0,0] = -5.0
print ">>> coads.sstcopy = pyferret.FerrPyVar(coads.sst._datagrid, datacopy, coads.sst._missingvalue, coads.sst._dataunit, 'modified copy of SST')"
coads.sstcopy = pyferret.FerrPyVar(coads.sst._datagrid, datacopy, coads.sst._missingvalue, coads.sst._dataunit, 'modified copy of SST')
print ">>> pyferret.showdatasets(brief=False)"
pyferret.showdatasets(brief=False)
print ">>> pyferret.setwindow(1, axisasp=0.5, logo=False)"
pyferret.setwindow(1,axisasp=0.5,logo=False)
print ">>> pyferret.setdefaulttext(font='Arial')"
pyferret.settextstyle(font='Arial')
print ">>> pyferret.shade(coads.sstcopy['40S':'40N','100E':'100W','16-FEB'])"
pyferret.shade(coads.sstcopy['40S':'40N','100E':'100W','16-FEB'])
print ">>> pyferret.saveplot('testferrpyvar.pdf')"
pyferret.saveplot('testferrpyvar.pdf')

print ">>> del coads.sstcopy"
del coads.sstcopy
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()
print ">>> coads.close()"
coads.close()
print ">>> del coads"
del coads
print ">>> pyferret.showdatasets()"
pyferret.showdatasets()

