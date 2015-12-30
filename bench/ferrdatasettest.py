# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

coads = pyferret.FerrDataSet('coads_climatology.cdf')
print "coads.filename = '%s'" % coads.filename
print 'repr(coads.varnames) = %s' % repr(coads.varnames)
print 'repr(coads) = %s' % repr(coads)
print 'str(coads) = %s' % str(coads)

