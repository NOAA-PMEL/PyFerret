# -*- coding: utf-8 -*-
"""
===========
ferretmagic
===========

Magics for interacting with ferret via pyferret.

.. note::

  The ``pyferret`` module needs to be installed first

Usage
=====

``%%ferret``

{ferret_DOC}

``%ferret_run``

{ferret_RUN_DOC}

``%ferret_getdata``

{ferret_GETDATA_DOC}

``%ferret_putdata``

{ferret_PUTDATA_DOC}

"""

#-----------------------------------------------------------------------------
#  Patrick.Brockmann@lsce.ipsl.fr
#  Started 2013/08/28 then put on github.com 2013/09/06
#
#-----------------------------------------------------------------------------

import sys
import os.path
import tempfile
from glob import glob
from shutil import rmtree

import numpy as np
import pyferret
from xml.dom import minidom

from IPython.core.displaypub import publish_display_data
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, needs_local_scope
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.utils.py3compat import unicode_to_str
from pexpect import ExceptionPexpect

_DEFAULT_PLOTSIZE = "720.0,612.0"
_DEFAULT_MEMSIZE = 50.0

#----------------------------------------------------
class ferretMagicError(Exception):
    pass

@magics_class
class ferretMagics(Magics):
    """A set of magics useful for interactive work with ferret via pyferret.

    """
#----------------------------------------------------
    def __init__(self, shell):
        """
        Parameters
        ----------
        shell : IPython shell

        """
        super(ferretMagics, self).__init__(shell)
        try:
            pyferret.start(journal=False, unmapped=True, quiet=True)
        except ExceptionPexpect:
            raise ferretMagicError('pyferret cannot be started')

#----------------------------------------------------
    def ferret_run_code(self, args, code):

        # Temporary directory
        temp_dir = tempfile.mkdtemp(prefix='ipyferret_').replace('\\', '/')
        txt_filename = temp_dir + '/output.txt' 
        if args.plotname:
            plot_filename = str(args.plotname)
        elif args.pdf:
            plot_filename = temp_dir + '/image.pdf'
        else:
            plot_filename = temp_dir + '/image.png'

        # Memory setting in double-precision "mega-words"
        if args.memory:
            mem_size = float(args.memory)
            if mem_size <= 0.0:
                mem_size = _DEFAULT_MEMSIZE
        else:
            mem_size = _DEFAULT_MEMSIZE
        (errval, errmsg) = pyferret.run('set memory/size=%f' % mem_size)

        # Plot size and aspect ratio
        if args.size:
            plot_size = args.size.split(',')
        else:
            plot_size = _DEFAULT_PLOTSIZE.split(',')
        plot_width  = float(plot_size[0])
        plot_height = float(plot_size[1])
        plot_aspect = float(plot_height) / float(plot_width)

        # Publish
        key = 'ferretMagic.ferret'

        #-------------------------------
        # Set window; use a standard-sized window and just set the aspect ratio
        if args.antialias:
            (errval, errmsg) = pyferret.run('set window /antialias /aspect=%(plot_aspect)f 1' % locals())
        else:
            (errval, errmsg) = pyferret.run('set window /noantialias /aspect=%(plot_aspect)f 1' % locals())

        # STDOUT handling
        (errval, errmsg) = pyferret.run('set redirect /clobber /file="%(txt_filename)s" stdout' % locals())
        # Cancel mode verify
        (errval, errmsg) = pyferret.run('cancel mode verify')
        # Run code
        pyferret_error = False
        for input in code:
            input = unicode_to_str(input)
            # ignore empty lines
            if input:
                (errval, errmsg) = pyferret.run(input)
                if errval != pyferret.FERR_OK:
                    publish_display_data(key, {'text/html': 
                        '<pre style="background-color:#F79F81; border-radius: 4px 4px 4px 4px; font-size: smaller">' +
                        'yes? %s\n' % input +
                        'error val = %i\nerror msg = %s' % (errval, errmsg) +
                        '</pre>' 
                    })
                    pyferret_error = True
                    break
            # Create image file; if no final image, no image file will be created
            # Any existing image with that filename will be versioned away ('.~n~' appended)
            if not pyferret_error:
                if args.pdf:
                    inch_width = plot_width / 72.0
                    (errval, errmsg) = pyferret.run('frame /xinch=%(inch_width)f /file="%(plot_filename)s" /format=PDF' % locals())
                else:
                    (errval, errmsg) = pyferret.run('frame /xpixel=%(plot_width)f /file="%(plot_filename)s" /format=PNG' % locals())
                if errval != pyferret.FERR_OK:
                    pyferret_error = True
            # Close stdout
            (errval, errmsg) = pyferret.run('cancel redirect')
            # Close window
            (errval, errmsg) = pyferret.run('cancel window 1')
            #-------------------------------

            # Publish
            display_data = []

            # Publish text output if not empty
            if os.path.getsize(txt_filename) != 0 : 
                try:
                    text_outputs = []
                    text_outputs.append('<pre style="background-color:#ECF6CE; border-radius: 4px 4px 4px 4px; font-size: smaller">')
                    f = open(txt_filename, "r")
                    for line in f:
                        text_outputs.append(line)
                    f.close()
                    text_outputs.append("</pre>")
                    text_output = "".join(text_outputs)
                    display_data.append((key, {'text/html': text_output}))
                except:
                    pass

            # Publish image if present
            if not pyferret_error:
               if args.pdf:
                   if os.path.isfile(plot_filename):
                       # Create link to pdf; file visible from cell from files directory
                       text_outputs = []
                       text_outputs.append('<pre style="background-color:#F2F5A9; border-radius: 4px 4px 4px 4px; font-size: smaller">')
                       text_outputs.append('Message: <a href="files/%(plot_filename)s" target="_blank">%(plot_filename)s</a> created.' % locals())
                       text_outputs.append('</pre>')
                       text_output = "".join(text_outputs)
                       display_data.append((key, {'text/html': text_output}))
                       # If the user did not provide the PDF filename, 
                       # do not delete the temporary directory since the PDF is in there.
                       if args.plotname:
                           rmtree(temp_dir)
                   else:
                       # Delete temporary directory - nothing to preserve
                       rmtree(temp_dir)
               else:
                   try:
                       f = open(plot_filename, 'rb')
                       image = f.read().encode('base64')
                       f.close()
                       display_data.append((key, {'text/html': '<div class="myoutput">' + 
                           '<img src="data:image/png;base64,%(image)s"/></div>' % locals()}))
                   except:
                       pass
                   # Delete temporary directory - PNG encoded in the string
                   rmtree(temp_dir)

        # Publication
        for source, data in display_data:
              publish_display_data(source, data)


#----------------------------------------------------
    @magic_arguments()
    @argument(
        '-m', '--memory', type=float,
        help='Physical memory used by ferret expressed in megawords. Default is %s megawords = %s megabytes.' % (str(_DEFAULT_MEMSIZE), str(8.0*_DEFAULT_MEMSIZE))
        )
    @argument(
        '-s', '--size',
        help='Pixel size of PNG plots or point size of PDF plots as "width,height". Default is ' + _DEFAULT_PLOTSIZE
        )
    @argument(
        '-a', '--antialias', default=False, action='store_true',
        help='Use anti-aliasing to improve the appearance of images and get smoother edges.' 
        )
    @argument(
        '-p', '--pdf', default=False, action='store_true',
        help='Generate the output plot as a PDF file.'
        )
    @argument(
        '-f', '--plotname',
        help='Name of the image file to create.  If not given, a name will be generated.'
        )
    @cell_magic
    def ferret(self, line, cell):
        '''
            In [10]: %%ferret
               ....: let a=12
               ....: list a

        The size of output plots can be specified:
            In [18]: %%ferret -s 800,600 
                ...: plot i[i=1:100]

        '''
        args = parse_argstring(self.ferret, line)
        code = cell.split('\n')
        self.ferret_run_code(args, code)

#----------------------------------------------------
    @magic_arguments()
    @argument(
        '-m', '--memory', type=float,
        help='Physical memory used by ferret expressed in megawords. Default is %s megawords = %s megabytes.' % (str(_DEFAULT_MEMSIZE), str(8.0*_DEFAULT_MEMSIZE))
        )
    @argument(
        '-s', '--size',
        help='Pixel size of PNG plots or point size of PDF plots as "width,height". Default is ' + _DEFAULT_PLOTSIZE
        )
    @argument(
        '-a', '--antialias', default=False, action='store_true',
        help='Use anti-aliasing technics to improve the appearance of images and get smoother edges.' 
        )
    @argument(
        '-p', '--pdf', default=False, action='store_true',
        help='Generate the output plot as a PDF file.'
        )
    @argument(
        '-f', '--plotname',
        help='Name of the image file to create.  If not given, a name will be generated.'
        )
    @argument(
        'string',
        nargs='*'
        )
    @line_magic
    def ferret_run(self, line):
        '''
        Line-level magic to run a command in ferret. 

            In [12]: for i in [100,500,1000]:
               ....:     %ferret_run -a -s 400,400 'plot sin(i[i=1:%(i)s]*0.1)' % locals()

        '''
        args = parse_argstring(self.ferret_run, line)
        code = [self.shell.ev(" ".join(args.string))]
        self.ferret_run_code(args, code)

#----------------------------------------------------
    @magic_arguments()
    @argument(
        '--create_mask', default=False, action='store_true',
        help='The data array associated with the "data" key will be a MaskedArray NumPy array instead an ordinary NumPy array.'
        )
    @argument(
        'code',
        nargs='*'
        )
    @line_magic
    def ferret_getdata(self, line):
        '''
        Line-level magic to get data from ferret.

            In [18]: %%ferret
               ....: use levitus_climatology
            In [19]: %ferret_getdata tempdict = temp
           ....: Message: tempdict is now available in python as a dictionary containing the variable's metadata and data array.
            In [20]: print tempdict.keys()
           ....: ['axis_coords', 'axis_types', 'data_unit', 'axis_units', 'title', 'axis_names', 'missing_value', 'data']

        '''

        args = parse_argstring(self.ferret_getdata, line)

        code = unicode_to_str(args.code[0])
        pythonvariable = code.split('=')[0]
        ferretvariable = code.split('=')[1]
        exec('%s = pyferret.getdata("%s", %s)' % (pythonvariable, ferretvariable, args.create_mask) )
        self.shell.push("%s" % pythonvariable)
        publish_display_data('ferretMagic.ferret', {'text/html': 
            '<pre style="background-color:#F2F5A9; border-radius: 4px 4px 4px 4px; font-size: smaller">' +
            'Message: ' + pythonvariable + " is now available in python as a dictionary containing the variable's metadata and data array."
            '</pre>' 
        })

#----------------------------------------------------
    @magic_arguments()
    @argument(
        '--axis_pos', default=None, 
        help='Order of the axes. Default mode uses a reasonable guess from examining the axis types.'
        )
    @argument(
        'code',
        nargs='*'
        )
    @line_magic
    def ferret_putdata(self, line):
        '''
        Line-level magic to put data to ferret.

            In [31]: import numpy as np
               ....: b = {}
               ....: b['name']='myvar'
               ....: b['name']='myvar'
               ....: x=np.linspace(-np.pi*4, np.pi*4, 500)
               ....: b['data']=np.sin(x)/x
               ....: b.keys()
            Out[31]: ['data', 'name']
        In [32]: %ferret_putdata --axis_pos (1,0,2,3,4,5) b
           ....: Message: b is now available in ferret as myvar

        '''
        args = parse_argstring(self.ferret_putdata, line)

        ferretvariable = unicode_to_str(args.code[0])
        if args.axis_pos:
            axis_pos_variable = eval(args.axis_pos)
        else:
            axis_pos_variable = None
        pyferret.putdata(self.shell.user_ns[ferretvariable], axis_pos=axis_pos_variable)
        publish_display_data('ferretMagic.ferret', {'text/html': 
            '<pre style="background-color:#F2F5A9; border-radius: 4px 4px 4px 4px; font-size: smaller">' +
            'Message: ' + ferretvariable + ' is now available in ferret as ' + self.shell.user_ns[ferretvariable]['name'] + 
            '</pre>' 
        })


#----------------------------------------------------
__doc__ = __doc__.format(
    ferret_DOC = ' '*8 + ferretMagics.ferret.__doc__,
    ferret_RUN_DOC = ' '*8 + ferretMagics.ferret_run.__doc__,
    ferret_GETDATA_DOC = ' '*8 + ferretMagics.ferret_getdata.__doc__,
    ferret_PUTDATA_DOC = ' '*8 + ferretMagics.ferret_putdata.__doc__
    )

def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(ferretMagics)

