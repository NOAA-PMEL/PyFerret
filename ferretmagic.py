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

``%ferret_lock``

{ferret_LOCK_DOC}

``%ferret_unlock``

{ferret_UNLOCK_DOC}

"""

#-----------------------------------------------------------------------------
#  Patrick.Brockmann@lsce.ipsl.fr
#  Started 2013/08/28 then put on github.com 2013/09/06
#  https://github.com/PBrockmann/ipython-ferretmagic
#
#  Lock functions are taken from ipythonPexpect magic
#  https://cdcvs.fnal.gov/redmine/projects/ipython_ext/repository/revisions/master/raw/ipythonPexpect.py
#-----------------------------------------------------------------------------

import os.path
import tempfile
import math
import pyferret
from shutil import rmtree

from IPython.core.displaypub import publish_display_data
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic, needs_local_scope
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring
from IPython.utils.py3compat import unicode_to_str
from pexpect import ExceptionPexpect

_PUBLISH_KEY = 'ferretMagic.ferret'
_DEFAULT_PLOTSIZE = '756.0,612.0'
_DEFAULT_MEMSIZE = 400.0

#----------------------------------------------------
class ferretMagicError(Exception):
    pass

@magics_class
class ferretMagics(Magics):
    """
    A set of magics useful for interactive work with ferret via pyferret.
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
            pyferret.start(memsize=(_DEFAULT_MEMSIZE/8.0), verify=False, journal=False, unmapped=True, quiet=True)
        except ExceptionPexpect:
            raise ferretMagicError('pyferret cannot be started')

        self._shell = shell
        self._shell.ferret_locked = False

#----------------------------------------------------
    def ferret_run_code(self, args, code):
        """
        Parameters
        ----------
        args : control arguments for running (py)ferret
        code : ferret commands to run
        """

        # Temporary directory; create under the current directory 
        # so PDF link files are accessible
        temp_dir = tempfile.mkdtemp(dir='.', prefix='ipyferret_').replace('\\', '/')

        # Redirect stdout and stderr to file
        out_filename = temp_dir + '/output.txt' 
        if not(args.quiet):
            (errval, errmsg) = pyferret.run('set redirect /clobber /file="%s" stdout' % out_filename)

        # Filename for saving the final plot (if any)
        if args.plotname:
            plot_filename = str(args.plotname)
            if args.pdf:
                if not plot_filename.endswith('.pdf'):
                    plot_filename += '.pdf'
            else:
                if not plot_filename.endswith('.png'):
                    plot_filename += '.png'
        elif args.pdf:
            plot_filename = temp_dir + '/image.pdf'
        else:
            plot_filename = temp_dir + '/image.png'


        # Make it quiet by default
        (errval, errmsg) = pyferret.run('cancel mode verify')

        if args.memory:
            # Reset memory size in megabytes
            mem_size = float(args.memory)
            if mem_size > 0.0:
                (errval, errmsg) = pyferret.run('set memory /size=%f' % (mem_size/8.0))

        # Get image size and aspect ratio
        if args.size:
            plot_size = args.size.split(',')
        else:
            plot_size = _DEFAULT_PLOTSIZE.split(',')
        plot_width  = float(plot_size[0])
        plot_height = float(plot_size[1])
        plot_aspect = plot_height / plot_width

        # Set window size with the required aspect ratio; 
        # always anti-alias with windows of these sizes
        canvas_width = math.sqrt(plot_width * plot_height / plot_aspect)
        if args.bigger:
            # Double the width and height of the window, but the image will
            # be saved at the original requested size.  
            # Reducing the raster image when saving it sharpens the image.
            canvas_width *= 2.0
        (errval, errmsg) = pyferret.run('set window /xpixel=%f /aspect=%f 1' % \
                                        (canvas_width, plot_aspect))

        # Run code
        pyferret_error = False
        for input in code:
            # Ignore blank lines
            if input:
                input = unicode_to_str(input)
                (errval, errmsg) = pyferret.run(input)
                if errval != pyferret.FERR_OK:
                    errmsg = errmsg.replace('\\', '<br />')
                    publish_display_data(_PUBLISH_KEY, {'text/html': 
                        '<pre style="background-color:#F79F81; border-radius: 4px 4px 4px 4px; font-size: smaller">' +
                        'yes? %s\n' % input +
                        '%s' % errmsg +
                        '</pre>' 
			})
                    pyferret_error = True
                    break

        # Create the image file; if no final image, no image file will be created.
        # Any existing image with that filename will be versioned away ('.~n~' appended)
        if not pyferret_error:
            if args.pdf:
                (errval, errmsg) = pyferret.run('frame /xinch=%f /file="%s" /format=PDF' % (plot_width/72.0, plot_filename) )
            else:
                (errval, errmsg) = pyferret.run('frame /xpixel=%f /file="%s" /format=PNG' % (plot_width, plot_filename))
            if errval != pyferret.FERR_OK:
                pyferret_error = True

        # Close the window
        (errval, errmsg) = pyferret.run('cancel window 1')

        # Close the stdout and stderr redirect file
        if not(args.quiet):
        	(errval, errmsg) = pyferret.run('cancel redirect')

        #-------------------------------

        # Publish
        display_data = []

        # Publish captured stdout text, if any
        if os.path.isfile(out_filename) and (os.path.getsize(out_filename) > 0): 
            try:
                text_outputs = []
                text_outputs.append('<pre style="background-color:#ECF6CE; border-radius: 4px 4px 4px 4px; font-size: smaller">')
                f = open(out_filename, "r")
                for line in f:
                    text_outputs.append(line)
                f.close()
                text_outputs.append("</pre>")
                text_output = "".join(text_outputs)
                display_data.append((_PUBLISH_KEY, {'text/html': text_output}))
            except:
                pass

        # Publish image if present
        if not pyferret_error:
           if args.pdf:
               if os.path.isfile(plot_filename):
                   # Create link to pdf; file visible from cell from files directory
                   text_outputs = []
                   text_outputs.append('<pre style="background-color:#F2F5A9; border-radius: 4px 4px 4px 4px; font-size: smaller">')
                   text_outputs.append('Message: <a href="files/%s" target="_blank">%s</a> created.' % (plot_filename, plot_filename))
                   text_outputs.append('</pre>')
                   text_output = "".join(text_outputs)
                   display_data.append((_PUBLISH_KEY, {'text/html': text_output}))
                   # If the user did not provide the PDF filename, 
                   # do not delete the temporary directory since the PDF is in there.
                   if args.plotname:
                       rmtree(temp_dir)
               else:
                   # Delete temporary directory - nothing to preserve
                   rmtree(temp_dir)
           else:
               # Display the image in the notebook
               try:
                   f = open(plot_filename, 'rb')
                   image = f.read().encode('base64')
                   f.close()
                   display_data.append((_PUBLISH_KEY, {'text/html': '<div class="myoutput">' + 
                       '<img src="data:image/png;base64,%s"/></div>' % image}))
               except:
                   pass
               # Delete temporary directory - PNG encoded in the string
               rmtree(temp_dir)

	# Error in ferret code - Delete temporary directory 
	else: 
           rmtree(temp_dir)

        # Publication
        for source, data in display_data:
              publish_display_data(source, data)


#----------------------------------------------------
    @magic_arguments()
    @argument(
        '-m', '--memory', type=float,
        help='Memory, in megabytes, to be used by ferret henceforth. Startup default is %s' % str(_DEFAULT_MEMSIZE)
        )
    @argument(
        '-s', '--size',
        help='Pixel size of PNG images, or point size of PDF images, as "width,height". Default is ' + _DEFAULT_PLOTSIZE
        )
    @argument(
        '-b', '--bigger', default=False, action='store_true',
        help='Produce a sharper plot by doubling the standard plot window size before scaling.'
        )
    @argument(
        '-p', '--pdf', default=False, action='store_true',
        help='Generate the output plot as a PDF file.'
        )
    @argument(
        '-q', '--quiet', default=False, action='store_true',
        help='Do not display stdout.'
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
        help='Memory, in megabytes, to be used by ferret henceforth. Startup default is %s' % str(_DEFAULT_MEMSIZE)
        )
    @argument(
        '-s', '--size',
        help='Pixel size of PNG images, or point size of PDF images, as "width,height". Default is ' + _DEFAULT_PLOTSIZE
        )
    @argument(
        '-b', '--bigger', default=False, action='store_true',
        help='Produce a sharper plot by doubling the standard plot window size before scaling.'
        )
    @argument(
        '-p', '--pdf', default=False, action='store_true',
        help='Generate the output plot as a PDF file.'
        )
    @argument(
        '-q', '--quiet', default=False, action='store_true',
        help='Do not display stdout.'
        )
    @argument(
        '-f', '--plotname',
        help='Name of the image file to create.  If not given, a name will be generated.'
        )
    @argument(
        'string',
        nargs='*'
        )
    @needs_local_scope
    @line_magic
    def ferret_run(self, line, local_ns=None):
        '''
        Line-level magic to run a command in ferret. 

            In [12]: for val in [100,500,1000]:
               ....:     %ferret_run -s 400,400 -b 'plot sin(i[i=1:%(val)s]*0.1)' % locals()

        '''
        args = parse_argstring(self.ferret_run, line)
        #code = [self.shell.ex(" ".join(args.string))]
        code = [eval(" ".join(args.string), local_ns)]
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

        code = unicode_to_str(''.join(args.code))
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

    @line_magic
    def ferret_lock(self, line):
        '''
        Lock the notebook to send EVERY executed cell through pyferret
      
        Do %ferret_unlock to unlock

        '''
    
        self._shell.ferret_locked = True

        print 'WARNING: All future cell execution will be processed through pyferret!'
        print 'To return to IPython, issue %ferret_unlock'

    @line_magic
    def ferret_unlock(self, line):
        '''
          Unlock the notebook to return to regular IPython
        '''

    
        self._shell.ferret_locked = False
    
        print 'Notebook will use IPython'

# Let's rewrite InteractiveShell.run_cell to do automatic processing with pyferret,
# if desired
from IPython.core.interactiveshell import InteractiveShell

# Let's copy the original "run_cell" method (we do this only once so we can reload)
if not getattr(InteractiveShell, "run_cell_a", False):
  InteractiveShell.run_cell_a = InteractiveShell.run_cell

# Now rewrite run_cell
def run_cell_new(self, raw_cell, store_history=False, silent=False, shell_futures=True):
  
  # Are we locked in pyferret?
  if getattr(self, "ferret_locked", False):
  
    # Don't alter cells that start with %%ferret or with %ferret_unlock
    if raw_cell[:8] == '%%ferret' or raw_cell[:15] == '%ferret_unlock':
      pass
    else:
      # We're going to add a %%ferret to the top
      raw_cell = "%%ferret\n" + raw_cell

  self.run_cell_a(raw_cell, store_history, silent, shell_futures)

# And assign it
InteractiveShell.run_cell = run_cell_new


#----------------------------------------------------
__doc__ = __doc__.format(
    ferret_DOC = ' '*8 + ferretMagics.ferret.__doc__,
    ferret_RUN_DOC = ' '*8 + ferretMagics.ferret_run.__doc__,
    ferret_GETDATA_DOC = ' '*8 + ferretMagics.ferret_getdata.__doc__,
    ferret_PUTDATA_DOC = ' '*8 + ferretMagics.ferret_putdata.__doc__,
    ferret_LOCK_DOC = ' '*8 + ferretMagics.ferret_lock.__doc__,
    ferret_UNLOCK_DOC = ' '*8 + ferretMagics.ferret_unlock.__doc__
    )

def load_ipython_extension(ip):
    """Load the extension in IPython."""
    ip.register_magics(ferretMagics)

