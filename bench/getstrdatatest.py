# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

pyferret.run('let strarr = {"one", "two", "three", "four", "five", "six"}')
pyferret.run('list strarr')
strdict = pyferret.getstrdata('strarr')
print 'string array metadata: '
print pyferret.metastr(strdict)
strdata = strdict['data']
print 'squeezed string array representation: '
print repr(strdata.squeeze())
del strdata
del strdict
strdict = pyferret.getstrdata('strarr')
print 'second fetch of the string array; dicitonary representation: '
print repr(strdict)
del strdict
