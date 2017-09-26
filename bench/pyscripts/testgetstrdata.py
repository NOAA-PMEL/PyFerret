# To be run in python after importing and starting pyferret
# such as from running "pyferret -python"

from __future__ import print_function

import sys ; sys.ps1 = '' ; sys.ps2 = ''
print()

print('>>> pyferret.run(\'let strarr = {"one", "two", "three", "four", "five", "six"}\')')
pyferret.run('let strarr = {"one", "two", "three", "four", "five", "six"}')
print(">>> pyferret.run('list strarr')")
pyferret.run('list strarr')
print(">>> strdict = pyferret.getstrdata('strarr')")
strdict = pyferret.getstrdata('strarr')
print(">>> print pyferret.metastr(strdict)")
print(pyferret.metastr(strdict))
print(">>> strdata = strdict['data']")
strdata = strdict['data']
print(">>> repr(strdata.squeeze())")
repr(strdata.squeeze())
print(">>> repr(strdict)")
repr(strdict)

