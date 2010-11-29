divert(-1)
define(`M4__SYSTEM', UNICOS)
# The following is not used because the Standard C header-file is bad
#define(`M4__STRING_DESCRIPTOR_INCLUDES',
#`#include <fortran.h>	for _fcd functions
#')
define(`M4__STRING_DESCRIPTOR_INCLUDES',
`#include "/usr//include//fortran.h"	for _fcd functions
')
define(`M4__FORTRAN_DEFINES',
`
#define FORTRAN_HAS_NO_BYTE
#define FORTRAN_HAS_NO_SHORT
')
# transformation from fortran name to name of C module
# for unicos, just convert to uppercase
define(`NAMEF',
       `translit($1,abcdefghijklmnopqrstuvwxyz,ABCDEFGHIJKLMNOPQRSTUVWXYZ)')
# transformation from string name to corresponding argument name
define(`STRINGF',`$1d')
# extra arguments, if any, for argument lengths
define(`STRINGX',`')
define(`REALX',`')
define(`INTEGERX',`')
define(`FUNCTIONX',`')
define(`DOUBLEX',`')
# declaration to be used for argument name descriptor
define(`STRINGD',`
    _fcd       $1d;') # declare string parameter as type _fcd
# declarations and initializations of canonical local variables
define(`STRINGL',`
    char      *$1	= _fcdtocp ($1d);
    unsigned   $1_len	= _fcdlen ($1d);')	# use _fcd functions
# FORTRAN declaration for a long integer (e.g. integer*4 for Microsoft)
define(`LONG_INT',`integer')
# FORTRAN declaration for a short integer (e.g. integer*2)
define(`SHORT_INT',`integer')
# FORTRAN declaration for an integer byte (e.g. integer*1 or byte)
define(`BYTE_INT',`integer')
# FORTRAN declaration for double precision (e.g. real for a Cray)
define(`DOUBLE_PRECISION',`real')
# FORTRAN syntax for including a file
define(`M4__RIGHT_QUOTE',')
define(`F_INCLUDE',`      `include' M4__RIGHT_QUOTE`'$1`'M4__RIGHT_QUOTE')
divert(0)dnl
