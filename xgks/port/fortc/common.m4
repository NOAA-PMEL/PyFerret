divert(-1)
define(`FC_NAME',`NAMEF($1)')
#
# diversion 1 is for collecting formal arguments
# diversion 2 is for extra formal arguments for string lengths
# diversion 3 is for formal argument declarations
# diversion 4 is for extra local variables derived from formal arguments
# diversion 5 is for prototype-style arguments
# diversion 6 is for extra prototype-style arguments
#
define(`STRING',`dnl
divert(1)ifdef(`INIT',,`, ')STRINGF(`$1')`'dnl
divert(2)STRINGX(`$1')`'dnl
divert(3)STRINGD(`$1')`'dnl
divert(4)STRINGL(`$1')`'dnl
divert(5)ifdef(`INIT',,`,')STRINGP(`$1')`'dnl
divert(6)STRINGPX(`$1')`'dnl
divert(0)undefine(`INIT')dnl
')dnl
#
#define(`STRING',`divert(1)ifdef(`INIT',,`, ')STRINGF(`$1')`'undefine(`INIT')divert(2)`'STRINGX(`$1')`'divert(3)`'STRINGD(`$1',`$2')`'divert(4)`'STRINGL(`$1')`'divert(5)`'translit(STRINGD(`$1',`$2'),`;',`,')divert(0)')dnl
#
define(`INTSTAR',`divert(1)ifdef(`INIT',,`, ')$1`'undefine(`INIT')divert(2)`'INTEGERX(`$1')`'divert(3)
    int       *$1;`'divert(0)')dnl
define(`FLOATSTAR',`divert(1)ifdef(`INIT',,`, ')$1`'undefine(`INIT')divert(2)`'REALX(`$1')`'divert(3)
    float     *$1;divert(0)')dnl
define(`DOUBLESTAR',`divert(1)ifdef(`INIT',,`, ')$1`'undefine(`INIT')divert(2)`'DOUBLEX(`$1')`'divert(3)
    double    *$1;divert(0)')dnl
define(`FUNCTION',`divert(1)ifdef(`INIT',,`, ')$2`'undefine(`INIT')divert(2)`'FUNCTIONX(`$2')`'divert(3)
    $1       (*$2)();divert(0)')dnl

# The following is for a pointer to a single character, not a Fortran 
# character variable
#
define(`CHARSTAR',`divert(1)ifdef(`INIT',,`, ')$1`'undefine(`INIT')divert(3)
    char      *$1;divert(0)')dnl

define(`VOIDSTAR',`divert(1)ifdef(`INIT',,`, ')$1`'undefine(`INIT')divert(3)
    void      *$1;divert(0)')dnl
define(`POINTER',`divert(1)ifdef(`INIT',,`, ')$2`'undefine(`INIT')divert(3)
    $1	**$2;divert(0)')dnl
define(`VARARGS',`undefine(`NOT_VARIADIC')')dnl
changecom()dnl
define(`M4__PROTO',`dnl
define(`INIT',1)dnl
define(`NOT_VARIADIC',1)dnl
$2`'dnl
ifdef(`NOT_VARIADIC',`dnl
NAMEF($1)(undivert(1)undivert(2))undivert(3)dnl
divert(-1)undivert(5)undivert(6)divert(0)',`dnl
#ifdef UD_STDARG
NAMEF($1)(undivert(5)undivert(6),
    ...)
#else
NAMEF($1)(undivert(1)undivert(2), va_alist)undivert(3)
    va_dcl
#endif
')')
changecom()
define(`M4__BODY',`
{undivert(4)')
divert(0)dnl
