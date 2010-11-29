divert(-1)
define(`M4__SYSTEM', VMS)
define(`M4__STRING_DESCRIPTOR_INCLUDES',
#include descrip
)
define(`M4__STRING_DESCRIPTOR_TYPEDEF',
/* VMS data structure for storing strings */
typedef struct dsc$descriptor_s *dscrp;
)
# transformation from fortran name to name of C module
define(`NAMEF',`$1')	# for vms, just use same name
# transformation from string name to corresponding argument name
define(`STRINGF',`$1d')	# append d for argument name descriptor
# extra arguments, if any, for argument lengths
define(`STRINGX',`')
define(`REALX',`')
define(`INTEGERX',`')
define(`FUNCTIONX',`')
define(`DOUBLEX',`')
# declaration to be used for argument name descriptor
define(`STRINGD',`
    dscrp      $1d;')	# declare argument string descriptors as type dscrp
# declarations and initializations of canonical local variables
define(`STRINGL',`
    char      *$1	= $1d->dsc$a_pointer;
    int        $1_len	= $1d->dsc$w_length;') # use descriptor components
# FORTRAN declaration for a long integer (e.g. integer*4 for Microsoft)
define(`LONG_INT',`integer')
# FORTRAN declaration for a short integer (e.g. integer*2)
define(`SHORT_INT',`integer*2')
# FORTRAN declaration for an integer byte (e.g. integer*1 or byte)
define(`BYTE_INT',`byte')
# FORTRAN declaration for double precision (e.g. real for a Cray)
define(`DOUBLE_PRECISION',`double precision')
# FORTRAN syntax for including a file
define(`M4__RIGHT_QUOTE',')
define(`F_INCLUDE',`      `include' M4__RIGHT_QUOTE`'$1`'M4__RIGHT_QUOTE')
divert(0)dnl
