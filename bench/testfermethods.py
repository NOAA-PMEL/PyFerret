# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print

print ">>> coads_uw  = pyferret.FerDSet('coads_uw.nc')"
coads_uw  = pyferret.FerDSet('coads_uw.nc')
print ">>> coads_uw.mywnd = (coads_uw.uwnd**2 + coads_uw.vwnd**2)**0.5"
coads_uw.mywnd = (coads_uw.uwnd**2 + coads_uw.vwnd**2)**0.5
print ">>> pyferret.showdata(brief=False)"
pyferret.showdata(brief=False)

print ">>> pyferret.setwindow(1, axisasp=0.75, color=(100,90,90), logo=False)"
pyferret.setwindow(1, axisasp=0.75, color=(100,90,90), logo=False)
print ">>> pyferret.settextstyle(font='arial', color='blue', bold=True, italic=True)"
pyferret.settextstyle(font='arial', color='blue', bold=True, italic=True)
print ">>> pyferret.run('SET REGION /T=15-FEB)"
pyferret.run('SET REGION /T=15-FEB')
print ">>> pyferret.contourplot(coads_uw.mywnd)"
pyferret.contourplot(coads_uw.mywnd)
print ">>> pyferret.saveplot('testfermethods_contour.pdf')"
pyferret.saveplot('testfermethods_contour.pdf')
print ">>> pyferret.fillplot(coads_uw.mywnd, line=True)"
pyferret.fillplot(coads_uw.mywnd, line=True)
print ">>> pyferret.saveplot('testfermethods_fill.pdf')"
pyferret.saveplot('testfermethods_fill.pdf')
print ">>> pyferret.shadeplot(coads_uw.mywnd)"
pyferret.shadeplot(coads_uw.mywnd)
print ">>> pyferret.saveplot('testfermethods_shade.pdf')"
pyferret.saveplot('testfermethods_shade.pdf')
print ">>> pyferret.run('SET REGION /Y=0 /T=15-FEB')"
pyferret.run('SET REGION /Y=0 /T=15-FEB')
print ">>> pyferret.lineplot(coads_uw.uwnd, vs=coads_uw.vwnd, color=coads_uw.mywnd)"
pyferret.lineplot(coads_uw.uwnd, vs=coads_uw.vwnd, color=coads_uw.mywnd)
print ">>> pyferret.saveplot('testfermethods_lineplot.pdf')"
pyferret.saveplot('testfermethods_lineplot.pdf')

print ">>> del coads_uw"
del coads_uw
print ">>> pyferret.showdata()"
pyferret.showdata()

