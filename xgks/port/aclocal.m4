define(diversion_number, divnum)dnl
divert(-1)


# Set the value of a variable.  Use the environment if possible; otherwise
# set it to a default value.  Call the substitute routine.
#
define([UC_DEFAULT], [dnl
$1=${$1-"$2"}
AC_SUBST([$1])
])


# Initialize this Unidata autoconf(1)-support module.
#
define([UC_INIT], [dnl
AC_INIT($1) dnl
UC_CUSTOMIZE dnl
UC_PORT dnl
])


# Set up for customizing the makefile in the port/ subdirectory.
#
define([UC_PORT], [dnl
AC_CONFIG_HEADER(port/udposix.h) dnl
UC_ENSURE(PORT_MANIFEST, udposix.h.in)dnl
UC_DEFAULT(CPPFLAGS, -DNDEBUG) dnl
UC_DEFAULT(CFLAGS, -O) dnl
UC_OS dnl
case "${OS}" in
  aix*)  UC_ENSURE(CPPFLAGS, -D_ALL_SOURCE);;
  hpux*) UC_ENSURE(CPPFLAGS, -D_HPUX_SOURCE);;
esac
UC_DEFAULT(LIBOBJS, ) dnl
UC_DEFAULT(PORT_HEADERS, ) dnl
UC_DEFAULT(PORT_MANIFEST, )
UC_DEFAULT(PORT_SUBDIRS, )
UC_PROG_CC dnl
UC_PROG_AR dnl
AC_PROG_RANLIB dnl
])


# Terminate this Unidata autoconf(1)-support module.
#
define([UC_TERM], [dnl
UC_CHECK_MISSING dnl
UC_POSTPROCESS_MAKEFILES($1) dnl
])


# Finish with everything (both GNU and Unidata autoconf(1) support).
#
define([UC_FINISH], [dnl
AC_OUTPUT($1)dnl
UC_TERM($1)dnl
])


# Handle a prerequsite m4(1) macro.
#
define([UC_REQUIRE], [ifdef([AC_PROVIDE_$1],,$1)])


# Check for functioning `const' keyword
#
define([UC_CONST], [dnl
AC_COMPILE_CHECK([working const], , [/* Ultrix mips cc rejects this.  */
typedef int charset[2]; const charset x;
], dnl
, dnl
[AC_DEFINE(UD_NO_CONST)])dnl
])


# Check for functioning `signed' keyword
#
define([UC_SIGNED],
[AC_COMPILE_CHECK([working signed], ,
changequote(,)dnl
signed char x;
changequote([,])dnl
, dnl
, dnl
[AC_DEFINE(UD_NO_SIGNED)])dnl
])


# Check for function prototypes
#
define([UC_PROTOTYPES],
[AC_COMPILE_CHECK([function prototypes], ,
extern int foo(int bar);
, dnl
, dnl
[AC_DEFINE(UD_NO_PROTOTYPES)])dnl
])


# Convert argument to uppercase.
#
define([UC_UPPERCASE],[translit($1,abcdefghijklmnopqrstuvwxyz,ABCDEFGHIJKLMNOPQRSTUVWXYZ)])


# Replace `/'s in argument with `_'s..
#
define([UC_SLASHTO_],[translit($1,/,_)])

# Return the C macro name version of the argument.
#
define([UC_C_MACRONAME], [UC_UPPERCASE(UC_SLASHTO_($1))])


# Obtain the pathname of a system-supplied header file.  The value of the
# associated shell variable is empty if the header-file could not be found.
#
define([UC_SYSTEM_HEADER], [dnl
AC_REQUIRE([UC_PROG_CPP])dnl
echo "#include <$1.h>" > conftestpath.c
dnl
dnl We add additional `/'s to the header file name to preclude compiler 
dnl warnings about the non-portability of `#include "/usr/include/..."'.
dnl
path=//`$CPP conftestpath.c 2> /dev/null |
    sed -n 's/^#.* 1 "\(.*$1\.h\)".*/\1/p' | 
    head -1`
rm -f conftestpath.c
AC_DEFINE(UD_SYSTEM_[]UC_C_MACRONAME(ifelse($2,,$1,$2))_H, \"$path\")
])


# Define macros for variadic function support
#
define([UC_VARIADIC_FUNCTIONS],[dnl
AC_REQUIRE([UC_PROG_CPP])dnl
UC_ENSURE(PORT_MANIFEST, stdarg.h.in)dnl
AC_COMPILE_CHECK([variadic function support], [#include <stdarg.h>],
;}
int foo(int bar, ...) {
    va_list     alist;
    va_start(alist, bar);
    bar = (int)va_arg(alist, int);
    va_end(alist);
    return bar;
, dnl
[UC_SYSTEM_HEADER(stdarg)], dnl
[AC_DEFINE(UD_NO_STDARG)
UC_SYSTEM_HEADER(varargs, stdarg)])dnl
UC_ENSURE(PORT_HEADERS, stdarg.h)dnl
AC_PROVIDE([$0])dnl
])


# Define macro for string generation
#
define([UC_MAKESTRING], [dnl
AC_COMPILE_CHECK([stringization], dnl
[# define MAKESTRING(x)	#x],
char *cp = MAKESTRING(foo);
, dnl
, dnl
[AC_DEFINE(UD_NO_STRINGIZATION)])dnl
])


# Define macro for token pasting.
#
define([UC_GLUE], [dnl
ifdef([AC_PROVIDE_$0], , [
AC_COMPILE_CHECK([token pasting], [#define GLUE(a,b) a ## b],
char *GLUE(c,p) = "foo";
, dnl
, dnl
[AC_DEFINE(UD_NO_TOKEN_PASTING)])
AC_PROVIDE([$0])dnl
])])


# Define pointer-to-void macro.
#
define([UC_VOIDP],
[AC_COMPILE_CHECK([void*], ,
extern void *foo();
, dnl
, dnl
[AC_DEFINE(UD_NO_VOIDSTAR)])])


# CFORTRAN support:
#
define([UC_CFORTRAN], [dnl
ifdef([AC_PROVIDE_$0], , [
echo "checking for cfortran.h"
UC_ENSURE(PORT_MANIFEST, cfortran_h)dnl
UC_ENSURE(PORT_HEADERS, cfortran.h)dnl
UC_REQUIRE([UC_GLUE])dnl
case "$DEFS" in
  *UD_NO_TOKEN_PASTING*) PORT_CFORTRAN=reiser;;
  *) PORT_CFORTRAN=stdc;;
esac
AC_SUBST(PORT_CFORTRAN)dnl
AC_PROVIDE([$0])dnl
])])


# Check for standard, udposix(3) stuff.
#
define([UC_UDPOSIX], [dnl
ifdef([AC_PROVIDE_$0], , [
AC_REQUIRE([UC_CONST])dnl
AC_REQUIRE([UC_SIGNED])dnl
AC_REQUIRE([UC_PROTOTYPES])dnl
AC_REQUIRE([UC_VARIADIC_FUNCTIONS])dnl
AC_REQUIRE([UC_MAKESTRING])dnl
AC_REQUIRE([UC_GLUE])dnl
AC_REQUIRE([UC_VOIDP])dnl
UC_ENSURE(PORT_MANIFEST, udposix.h.in)dnl
AC_PROVIDE([$0])dnl
])])


# Check for a function.
#
define([UC_FUNC],
[AC_COMPILE_CHECK(function $2 declaration, [#include $1], dnl
extern struct {int foo;} *$2();
, dnl
[AC_DEFINE(UD_NO_[]UC_UPPERCASE($2)_DECL)
AC_REPLACE_FUNCS($2)dnl
])])


# Check for a type definition.
#
define([UC_TYPEDEF],
[AC_COMPILE_CHECK(typedef $2, [#include $1
typedef void $2;], , [AC_DEFINE(UD_NO_[]UC_UPPERCASE($2))])dnl
])


# Check for a structure definition.
#
define([UC_STRUCT], [dnl
AC_COMPILE_CHECK(structure $2, [#include $1
struct $2 {char *foo;};], , [AC_DEFINE(UD_NO_[]UC_UPPERCASE($2)[]_STRUCT)])dnl
])


# Ensure a macro definition.
#
define([UC_MACRO], [dnl
AC_COMPILE_CHECK(macro $2, [#include $1
#ifdef $2
  #error
#endif], , [AC_DEFINE(UD_NO_[]UC_UPPERCASE($2)_MACRO)])dnl
])


# Ensure a POSIX <limits.h>.
#
define([UC_UDPOSIX_LIMITS], [dnl
ifdef([AC_PROVIDE_$0], , [dnl
AC_REQUIRE([UC_PROG_CC])dnl
UC_ENSURE(PORT_MANIFEST, config.c)dnl
AC_HEADER_CHECK(limits.h, UC_SYSTEM_HEADER(limits), [dnl
UC_ENSURE(PORT_HEADERS, limits.h)], dnl
[UC_ENSURE(PORT_HEADERS)])dnl
])])


# Ensure a POSIX <float.h>.
#
define([UC_UDPOSIX_FLOAT], [dnl
ifdef([AC_PROVIDE_$0], , [dnl
AC_REQUIRE([UC_PROG_CC])dnl
echo "checking for conforming <float.h>"
UC_ENSURE(PORT_MANIFEST, config.c)dnl
AC_TEST_CPP([#include <float.h>
#define DBL_DIG foobar], dnl
[UC_ENSURE(PORT_HEADERS)], dnl
[UC_ENSURE(PORT_HEADERS, float.h)])dnl
])])


# Ensure a POSIX <stdarg.h>.
#
define([UC_UDPOSIX_STDARG], [dnl
AC_REQUIRE([UC_VARIADIC_FUNCTIONS])dnl
])


# Ensure a POSIX <stddef.h>.
#
define([UC_UDPOSIX_STDDEF], [dnl
ifdef([AC_PROVIDE_$0], , [dnl
AC_REQUIRE([UC_UDPOSIX])dnl
UC_ENSURE(PORT_MANIFEST, stddef.h.in)dnl
UC_ENSURE(PORT_HEADERS, stddef.h)dnl
UC_SYSTEM_HEADER(stddef)dnl
UC_MACRO(<stddef.h>, offsetof, (type\, member), dnl
    ((size_t)\&((type*)0)->member))dnl
AC_PROVIDE([$0])dnl
])])


# Ensure a POSIX <stdlib.h>.  NB: Don't check for `size_t' because, by
# convention, all portable programs will include <stddef.h> before
# <stdlib.h> so that any compiler-supplied size_t declaration lockout
# mechanisms will be activated.
#
define([UC_UDPOSIX_STDLIB], [dnl
AC_REQUIRE([UC_UDPOSIX])dnl
UC_ENSURE(PORT_MANIFEST, stdlib.h.in atexit.c)dnl
UC_ENSURE(PORT_HEADERS, stdlib.h)dnl
UC_SYSTEM_HEADER(stdlib)dnl
dnl UC_TYPEDEF(<stdlib.h>, div_t, struct div { int quot; int rem; })dnl
dnl UC_TYPEDEF(<stdlib.h>, ldiv_t, struct ldiv { long quot; long rem; })dnl
dnl UC_TYPEDEF(<stdlib.h>, size_t, unsigned int)dnl
UC_FUNC(<stdlib.h>, atexit, int atexit, (void (*fcn)(void)))dnl
AC_HAVE_FUNCS(on_exit)dnl
AC_PROVIDE([$0])dnl
])


# Ensure a POSIX <string.h>.  NB: Don't check for `size_t' because, by
# convention, all portable programs will include <stddef.h> before
# <string.h> so that any compiler-supplied size_t declaration lockout
# mechanisms will be activated.
#
define([UC_UDPOSIX_STRING], [dnl
AC_REQUIRE([UC_UDPOSIX])dnl
UC_ENSURE(PORT_MANIFEST, strerror.c string.h strstr.c string.h.in)dnl
UC_ENSURE(PORT_HEADERS, string.h)dnl
UC_SYSTEM_HEADER(string)dnl
dnl UC_TYPEDEF(<string.h>, size_t, unsigned int)dnl
UC_FUNC(<string.h>, strerror, char *strerror, (int errno))dnl
UC_FUNC(<string.h>, strstr, char *strstr, 
    (const char *cs\, const char *ct))dnl
AC_HAVE_FUNCS(bcopy [[index]] rindex)dnl
AC_PROVIDE([$0])dnl
])


# Ensure a POSIX <time.h>.
#
define([UC_UDPOSIX_TIME], [dnl
ifdef([AC_PROVIDE_$0], , [
AC_REQUIRE([UC_UDPOSIX])dnl
UC_ENSURE(PORT_MANIFEST, difftime.c time.h.in)dnl
UC_ENSURE(PORT_HEADERS, time.h)dnl
UC_SYSTEM_HEADER(time)dnl
UC_TYPEDEF(<time.h>, time_t, long)dnl
UC_FUNC(<string.h>, difftime, double difftime, (time_t t1, time_t t0))dnl
AC_PROVIDE([$0])dnl
])])


# Ensure a POSIX <signal.h>.
#
define([UC_UDPOSIX_SIGNAL], [dnl
AC_REQUIRE([UC_UDPOSIX])dnl
UC_ENSURE(PORT_MANIFEST, signal.h.in sigaddset.c \
    sigdelset.c sigemptyset.c sigprocmask.c sigsuspend.c)dnl
UC_ENSURE(PORT_HEADERS, signal.h)dnl
UC_SYSTEM_HEADER(signal)dnl
UC_TYPEDEF(<signal.h>, sigset_t, unsigned long)dnl
UC_TYPEDEF(<signal.h>, sig_atomic_t, int)dnl
UC_STRUCT(<signal.h>, sigaction, {void (*sa_handler)(); sigset_t sa_mask; int sa_flags;})dnl
UC_FUNC(<signal.h>, sigaction, int sigaction, (int sig\, const struct sigaction *act\, struct sigaction * oact))dnl
UC_FUNC(<signal.h>, sigemptyset, int sigemptyset, (sigset_t *set))dnl
UC_FUNC(<signal.h>, sigfillset, int sigfillset, (sigset_t *set))dnl
UC_FUNC(<signal.h>, sigaddset, int sigaddset, (sigset_t *set\, int signo))dnl
UC_FUNC(<signal.h>, sigdelset, int sigdelset, (sigset_t *set\, int signo))dnl
UC_FUNC(<signal.h>, sigismember, int sigismember, (const sigset_t *set\, int signo))dnl
UC_FUNC(<signal.h>, sigaction, int sigaction, (int sig\, const struct sigaction *act\, struct sigaction *oact))dnl
UC_FUNC(<signal.h>, sigprocmask, int sigprocmask, (int how\, const sigset_t *set\, sigset_t *oset))dnl
UC_FUNC(<signal.h>, sigpending, int sigpending, (sigset_t *set))dnl
UC_FUNC(<signal.h>, sigsuspend, int sigsuspend, (const sigset_t *set))dnl
AC_HAVE_FUNCS(sigvec sigblock sigpause sigsetmask sigstack bsdsigp)dnl
AC_PROVIDE([$0])dnl
])


# Ensure a <select.h>.
#
define([UC_SELECT], [dnl
UC_ENSURE(PORT_MANIFEST, select.h)dnl
UC_SYSTEM_HEADER(sys/select)dnl
])


# Check for C compiler.  This macro replaces the AC_PROG_CC macro because 
# that macro prefers the GNU C compiler.  Note that `c89' isn't checked for;
# this is because that compiler hides things like NBBY.
#
define([UC_PROG_CC], [dnl
ifdef([AC_PROVIDE_$0], , [
AC_BEFORE([$0], [UC_PROG_CPP])dnl
AC_PROGRAM_CHECK(CC, cc, cc, )dnl
if test -z "$CC"; then
  UC_NEED_VALUE(CC, [C compiler], /bin/cc)dnl
fi
# Find out if we are using GNU C, under whatever name.
cat <<EOF > conftest.c
#ifdef __GNUC__
  yes
#endif
EOF
${CC-cc} -E conftest.c > conftest.out 2>&1
if egrep yes conftest.out >/dev/null 2>&1; then
  GCC=1 # For later tests.
  CC="$CC -O"
fi
rm -f conftest*
AC_PROVIDE([$0])dnl
])])


# Check for cpp(1).  This macro replaces the AC_PROG_CPP macro because:
#	1. That macro, for some reason, sets the value of the shell 
#	   variable `CPP' to `${CC-cc} -E' rather than to the cpp(1)
#	   program and such a value has caused trouble in shell command
#	   lines;
#	2. The documentation explicitly states that the AC_PROG_CPP macro 
#	   should be called after the AC_PROG_CC macro, so there's no reason 
#	   for the above value that I can see; and
#	3. We need to discover when ${CPP} doesn't work (e.g. when it's 
#	   defined as `acc -E' under older versions of SunOS).
#
define([UC_PROG_CPP], [dnl
ifdef([AC_PROVIDE_$0], , [
UC_REQUIRE([UC_PROG_CC])dnl
AC_PROG_CPP[]dnl
CPP=`eval echo $CPP`
echo "#include <stdlib.h>" > conftest.c
if test `$CPP conftest.c 2> /dev/null | wc -l` = 0; then
  if test "$CPP" = cpp; then
    echo 1>&2 "$[]0: C preprocessor, \`$CPP', doesn't work"
    UC_NEED_VALUE(CPP, [C preprocessor], /lib/cpp)dnl
  else
    echo 1>&2 "$[]0: C preprocessor, \`$CPP', doesn't work; setting to \`cpp'"
    CPP=cpp
    if test `which ${CPP} 2>&1 | wc -w` != 1; then
      echo 1>&2 "$[]0: C preprocessor, \`$CPP', doesn't exist"
      UC_NEED_VALUE(CPP, [C preprocessor], /lib/cpp)dnl
    fi
  fi
fi
rm -f conftest.c
AC_PROVIDE([$0])dnl
])])


# Check for FORTRAN compiler.
#
define([UC_PROG_FC], [dnl
ifdef([AC_PROVIDE_$0], , [ dnl
UC_REQUIRE([UC_OS])dnl
case "$OS" in
  hpux*) AC_PROGRAMS_CHECK(FC, fort77 fortc f77);;
  *)     AC_PROGRAMS_CHECK(FC, f77 cf77);;
esac
if test -z "$FC"; then
  UC_NEED_VALUE(FC, [FORTRAN compiler], /bin/f77)dnl
fi
AC_PROVIDE([$0])dnl
])])


# Check for FORTRAN library.
#
define([UC_LIB_F77], [dnl
ifdef([AC_PROVIDE_$0], , [ dnl
AC_REQUIRE([UC_PROG_FC])dnl
echo checking for FORTRAN library
case `which "$FC"` in
  *lang*) LD_F77='-lF77 -lM77';;
  *)	  LD_F77='-lF77';;
esac
AC_SUBST(LD_F77)dnl
AC_PROVIDE([$0])dnl
])
])


# Check for library utility, ar(1).
#
define([UC_PROG_AR], [dnl
AC_PROGRAM_CHECK(AR, ar, ar, )dnl
if test -z "$AR"; then
  UC_NEED_VALUE(AR, [library utility], /bin/ar)dnl
fi
AC_PROVIDE([$0])dnl
])


# Check for troff(1).
#
define([UC_PROG_TROFF], [dnl
AC_PROGRAM_CHECK(TROFF, troff, ptroff, troff)dnl
if test -z "$TROFF"; then
  UC_NEED_VALUE(TROFF, [troff(1)-like utility], /bin/troff)dnl
fi
AC_PROVIDE([$0])dnl
])


# Check for fortc(1)
#
define([UC_PROG_FORTC], [dnl
AC_REQUIRE([UC_OS])dnl
AC_REQUIRE([UC_UDPOSIX_STDDEF])dnl
UC_ENSURE(PORT_SUBDIRS, fortc)dnl
UC_ENSURE(PORT_MANIFEST, fortc.h fortc.fc udalloc.h)dnl
dir=`pwd`/port/fortc
FORTC="$dir/fortc"
AC_SUBST(FORTC)dnl
NEED_FORTC=yes
AC_SUBST(NEED_FORTC)dnl
])


# Check for neqn(1).
#
define([UC_PROG_NEQN], [dnl
AC_PROGRAM_CHECK(NEQN, neqn, neqn, cat)dnl
test "$NEQN" = cat && 
  echo 1>&2 "$[]0: Can't find program \`neqn'; setting to \`cat'"
])


# Check for tbl(1).
#
define([UC_PROG_TBL], [dnl
AC_PROGRAM_CHECK(TBL, tbl, tbl, cat)dnl
test "$TBL" = cat && 
  echo 1>&2 "$[]0: Can't find \`tbl'; setting to \`cat'"
])


# Determine the operating system.
#
define([UC_OS], [dnl
ifdef([AC_PROVIDE_$0], , [
if test -z "$OS"; then
echo checking for type of operating system
cat << \CAT_EOF > conftest.c
#ifdef __osf__
OS_osf
#endif
#ifdef _AIX
OS_aix
#endif
#ifdef hpux
OS_hpux
#endif
#ifdef sgi
OS_irix
#endif
#ifdef sun
OS_sunos
#endif
#ifdef ultrix
OS_ultrix
#endif
#ifdef _UNICOS
OS_unicos
#endif
CAT_EOF
OS=`cc -E conftest.c | sed -n '/^OS_/ {
  s///p
  q
}'`
rm conftest.c
if test -z "$OS"; then
  UC_NEED_VALUE(OS, [operating system], sunos)dnl
fi
fi
AC_SUBST(OS)dnl
AC_PROVIDE([$0])dnl
])])


# Check for ncdump(1)
#
define([UC_PROG_NCDUMP], [dnl
AC_PROGRAM_CHECK(NCDUMP, ncdump, ncdump, UC_ABSPATH($exec_prefix)/ncdump)dnl
if test `which "$NCDUMP" | wc -w` != 1; then
  UC_NEED_VALUE(NCDUMP, [netCDF lister], /usr/local/unidata/bin/ncdump)dnl
fi
])


# Check for ncgen(1)
#
define([UC_PROG_NCGEN], [dnl
AC_PROGRAM_CHECK(NCGEN, ncgen, ncgen, UC_ABSPATH($exec_prefix)/ncgen)dnl
if test `which "$NCGEN" | wc -w` != 1; then
  UC_NEED_VALUE(NCGEN, [netCDF generator], /usr/local/unidata/bin/ncgen)dnl
fi
])


# Test a script.
#
define([UC_TEST_SCRIPT],
[cat << EOF > conftest.sh
[$1]
EOF
chmod +x conftest.sh
if ./conftest.sh 2> /dev/null; then
  ifelse([$2], , :, [$2])
ifelse([$3], , , [else
  $3
])dnl
fi
rm -f conftest.sh
])dnl


# Filter a file through cpp(1).
#
define([UC_FILTER_CPP], [dnl
AC_REQUIRE([UC_PROG_CPP])dnl
echo processing $1 with the C preprocessor to produce $2
ifdef([AC_CONFIG_NAME],
UC_TEST_SCRIPT([dnl
echo "$DEFS" > conftest.c
echo "# line 1 $1" >> conftest.c
cat $1 >> conftest.c
$CPP conftest.c | \
    awk '/^$/ {if (set) next; set=1} {print} !/^$/ {set=0}' > $2
rm -f conftest.c]), dnl
[UC_TEST_SCRIPT(
[$CPP "$DEFS" $1 | \
    awk '/^$/ {if (set) next; set=1} {print} !/^$/ {set=0}' > $2])])])


# Convert a pathname to an absolute one at autoconf(1) execution time.
#
define([UC_ABSPATH_M4], [dnl
syscmd([case "$1" in 
  /*) echo $1; exit;;
   *) path=`pwd`/$1
      tail=
      while test -n "$path"; do
        (cd $path && echo `pwd`$rest) 2> /dev/null && exit
        base=/`basename "$path"`
        tail=/$base$tail
        path=`echo "$path" | sed "s/\/$base//"`
      done;;
esac > conftest.syscmd 2>&1
])dnl
include(conftest.syscmd)dnl
])


# Convert a pathname to an absolute one at ./configure execution time.
#
define([UC_ABSPATH], [`dnl
case "$1" in 
  /*[)] echo $1; exit;;
   *[)] path=\`pwd\`/$1
        tail=
        while test -n "$path"; do
          (cd $path && echo \`pwd\`$rest) 2> /dev/null && exit
          base=/\`basename "$path"\`
          tail=/$base$tail
          path=\`echo "$path" | sed "s/\/$base//"\`
        done;;
esac
`])


# Set a value for the installation prefix.
#
define([UC_PREFIX], 
[AC_BEFORE([$0],[UC_PROG_FORTC])dnl
AC_BEFORE([$0],[UC_LIB_NETCDF])AC_BEFORE([$0],[UC_CPP_NETCDF])dnl
AC_BEFORE([$0],[UC_LIB_NCOPERS])AC_BEFORE([$0],[UC_CPP_NCOPERS])dnl
AC_BEFORE([$0],[UC_LIB_UDPORT])dnl
echo setting the installation prefix
prefix=UC_ABSPATH(${prefix-$1})
AC_SUBST([prefix])dnl
test -z "$exec_prefix" && exec_prefix=$prefix/bin
AC_SUBST([exec_prefix])dnl
AC_PROVIDE([$0])dnl
])


# Check for a directory containing a file.
#
define([UC_TEST_DIR], [dnl
  if test -z "$$1"; then
    for dir in $2; do
      if test -r $dir/$3; then
        $1=$dir
        break;
      fi
    done
    if test -z "$$1"; then
      UC_NEED_VALUE($1, $4, $5)dnl
    fi
  fi
AC_SUBST($1)dnl
])


# Check for X11 header-file directory.
#
define([UC_CPP_X11], [dnl
echo checking for X11 header-files
UC_TEST_DIR(CPP_X11, ${OPENWINHOME-/usr/openwin}/[[include]] \
    /usr/[[include]] /usr/local/[[include]], X11/Xlib.h,
    X11 [[[include]]]-directory, -I/usr/openwin/[[[include]]])dnl
CPP_X11=`case ${CPP_X11} in -I*) echo ${CPP_X11};; *) echo -I${CPP_X11-};; esac`
AC_PROVIDE([$0])dnl
])


# Check for McIDAS library.
#
define([UC_LIB_MCIDAS], [dnl
echo checking for MCIDAS library
UC_TEST_LIB(LD_MCIDAS, /home/mcidas/lib /home/mcidasd/lib, mcidas, McIDAS, dnl
  -L/home/mcidas/lib -lmcidas)dnl
AC_PROVIDE([$0])dnl
])


# Check for X11 library.
#
define([UC_LIB_X11], [dnl
echo checking for X11 library
UC_TEST_LIB(LD_X11, ${OPENWINHOME-/usr/openwin}/lib /usr/lib dnl
  /usr/lib/X11 /usr/local/lib /usr/local/lib/X11, X11, X11, dnl
  -L/usr/lib/X11 -lX11)dnl
AC_PROVIDE([$0])dnl
])


# Check for X11 implementation (header file and library).
#
define([UC_X11], [AC_REQUIRE([UC_CPP_X11])AC_REQUIRE([UC_LIB_X11])])


# Check for netCDF header-file directory.
#
define([UC_CPP_NETCDF], [dnl
echo checking for netCDF header-file
UC_TEST_DIR(CPP_NETCDF, UC_ABSPATH($prefix/[[[include]]]), netcdf.h,
    [netCDF [[include]]-directory], [-I/usr/local/unidata/[[include]]])dnl
CPP_NETCDF=`case ${CPP_NETCDF} in -I*) echo ${CPP_NETCDF};; *) echo -I${CPP_NETCDF-};; esac`
AC_PROVIDE([$0])dnl
])


# Check for netCDF library.
#
define([UC_LIB_NETCDF], [dnl
echo checking for netCDF library
UC_TEST_LIB(LD_NETCDF, UC_ABSPATH($prefix/lib), netcdf,
  netCDF, -L/usr/local/unidata/lib -lnetcdf)dnl
AC_PROVIDE([$0])dnl
])


# Check for netCDF implementation (header file and library).
#
define([UC_NETCDF], [AC_REQUIRE([UC_CPP_NETCDF])AC_REQUIRE([UC_LIB_NETCDF])])


# Check for netCDF operators library.
#
define([UC_LIB_NCOPERS], [dnl
echo checking for netCDF operators library
UC_TEST_LIB(LD_NCOPERS, UC_ABSPATH($prefix/lib), ncopers,
  netCDF-operators, [-L/usr/local/unidata/lib -lncopers])dnl
AC_PROVIDE([$0])dnl
])


# Check for LDM header-file directory.
#
define([UC_CPP_LDM], [dnl
echo checking for LDM header-file
UC_TEST_DIR(CPP_LDM, UC_ABSPATH($prefix/[[[include]]]) dnl
    UC_ABSPATH($prefix/../[[[include]]]) dnl
    UC_ABSPATH($prefix/../ldm/[[[include]]]), ldm.h,
    [LDM [[include]]-directory], [-I/usr/local/unidata/[[include]]])dnl
CPP_LDM=`case ${CPP_LDM} in -I*) echo ${CPP_LDM};; *) echo -I${CPP_LDM-};; esac`
if test -z "$CPP_LDM"; then
  UC_NEED_VALUE(CPP_LDM, [LDM include directory], -I/home/ldm/include)dnl
fi
AC_PROVIDE([$0])dnl
])


# Check for LDM library.
#
define([UC_LIB_LDM], [dnl
echo checking for LDM library
UC_TEST_LIB(LD_LDM, UC_ABSPATH($prefix/lib) dnl
  UC_ABSPATH($prefix/../lib) UC_ABSPATH($prefix/../ldm/lib), ldm,
  LDM, -L/usr/local/unidata/lib -lldm)dnl
if test -z "$LD_LDM"; then
  UC_NEED_VALUE(LD_LDM, [LDM library], -L/home/ldm/lib -lldm)dnl
fi
AC_PROVIDE([$0])dnl
])


# Check for LDM implementation (header file and library).
#
define([UC_LDM], [AC_REQUIRE([UC_CPP_LDM])AC_REQUIRE([UC_LIB_LDM])])


# Check for udres(3) library.
#
define([UC_LIB_UDRES], [dnl
echo 'checking for udres library'
UC_TEST_LIB(LD_UDRES, UC_ABSPATH($prefix/lib), udape,
  udres, -L/usr/local/unidata/lib -ludape)dnl
AC_PROVIDE([$0])dnl
])


# Set installation programs.  This differs from the standard
# autoconf(1) macro by making installed data files group writable.
#
define([UC_PROG_INSTALL], [dnl
AC_PROG_INSTALL[]dnl
INSTALL_DATA="`echo "${INSTALL_DATA}" | sed 's/644/664/'`"
AC_PROVIDE([$0])dnl
])


# Check for a library.  It would have been nice to see if a compile-link-
# execute sequence would have worked (via AC_TEST_PROGRAM) but, with dynamic
# libraries under SunOS, the link and execution fail due to unresolved 
# references.  Ergo, we just check for the regular `.a' file.
#
define([UC_TEST_LIB], [dnl
if test -z "$$1"; then
  for dir in $2; do
    if test -r $dir/lib$3.a; then
      $1="-L$dir -l$3"
      break
    fi
  done
  if test -z "$$1"; then
      UC_NEED_VALUE($1, [$4 library], $5)dnl
  fi
fi
AC_SUBST($1)dnl
])


# Check for missing definitions.
#
define([UC_CHECK_MISSING], [dnl
if test -s conf.missing; then
  cat << CAT_EOF

$[]0: The following variables need values.  They may be set in
the environment or in the file CUSTOMIZE.  Variables referring to
executable programs needn't be set if the relevant directory is added to
PATH.  In any case, ./configure should probably be rerun.  See file INSTALL
for details.

CAT_EOF
  awk -F: 'BEGIN {printf "%-13s%-27s%s\n", "VARIABLE", "MEANING", "EXAMPLE";
	          printf "%-13s%-27s%s\n", "--------", "-------", "-------"}
	         {printf "%-13s%-27s%s\n", $[]1, $[]2, $[]3}' conf.missing
  rm conf.missing
  exit 1
fi
rm -f conf.missing
])


# Post process makefiles.
#
define([UC_POSTPROCESS_MAKEFILES], [dnl
AC_REQUIRE([UC_PROG_CC])dnl
define(dn,divnum)divert(-1)undivert[]divert(dn)undefine([dn])dnl
# Post process any makefiles.
#
# Create a script to accomplish the post processing.
#
cat << \EOF_CONFTEST_SH > conftest.sh
cat << \EOF_CONFTEST_C > conftest.c
#include <stdio.h>
main()
{
    return readsub((char*)NULL) ? 0 : 1;
}
readsub(inpath)
    char	*inpath;
{
    char	buf[[2048]], path[[1024]];
    FILE	*fp	= inpath == NULL
				? stdin
				: fopen(inpath, "r");
    if (fp == NULL) {
	(void) perror(inpath);
	return 0;
    }
    buf[[sizeof(buf)-1]]	= 0;
    while (fgets(buf, sizeof(buf), fp) != NULL) {
	if (sscanf(buf, "[include]%*[[] \t[]]%s", path) == 1) {
	    if (!readsub(path))
		return 0;
	} else {
	    (void) fputs(buf, stdout);
	}
    }
    return 1;
}
EOF_CONFTEST_C
if $CC -o conftest conftest.c; then
    conftest=`pwd`/conftest
    set $1
    for file do
      echo post processing makefile \`$file\'
      sd=`pwd`/`echo $file | sed 's,[[^/]]*$,,'`
      base=`basename $file`
      (cd $sd; $conftest < $base > conftest.mk && mv conftest.mk $base)
    done
fi
rm conftest conftest.c
EOF_CONFTEST_SH
#
# Append the above script to the output-script file, config.status, so that 
# invoking that file will also do the post processing.  Note that the 
# output-script file will be invoked by ./configure before the post-
# processing code is appended.
#
cat conftest.sh >> config.status
#
# If appropriate, do the postprocessing now because the previous step 
# couldn't.
#
test -n "$no_create" || CC=$CC sh conftest.sh
rm conftest.sh
])


# Get shell-variable override values for local customizations.
#
define([UC_CUSTOMIZE], [dnl
AC_BEFORE([$0], [UC_DEFAULT])dnl
if [[ -r CUSTOMIZE ]]; then
  echo reading configuration customizations
  . ./CUSTOMIZE
fi
])


# Set the root of the FTP distribution directory.
#
define([UC_FTPDIR], [dnl
FTPDIR=${FTPDIR-/home/ftp}/$1
AC_SUBST(FTPDIR)dnl
])


# Set package name.
#
define([UC_PACKAGE], [dnl
echo checking for package name
PACKAGE=${PACKAGE-`basename \`pwd\``}
AC_SUBST(PACKAGE)dnl
])


# Set version identifiers.
#
define([UC_VERSION], [dnl
echo checking for package version
if test -z "$VERSION"; then
  if test -r VERSION; then \
    VERSION="`cat VERSION`"
  else
    VERSION=
  fi
fi
AC_SUBST(VERSION)dnl
if test -z "$MAJOR_NO"; then
  if test -n "$VERSION"; then \
    MAJOR_NO=`echo $VERSION |
      sed -n '/^\([[0-9]][[0-9]]*\)\.[[0-9]][[0-9]]*.*/s//\1/p;q'`
  else
    MAJOR_NO=
  fi
fi
AC_SUBST(MAJOR_NO)dnl
if test -z "$MINOR_NO"; then
  if test -n "$VERSION"; then \
    MINOR_NO=`echo $VERSON |
      sed -n '/^[[0-9]][[0-9]]*\.\([[0-9]][[0-9]]*\).*/s//\1/p;q'`
  else
    MINOR_NO=
  fi
fi
AC_SUBST(MINOR_NO)dnl
])


# Handle a missing value.
#
define([UC_NEED_VALUE], [dnl
echo "$1:$2:$3" >> conf.missing
])


# Ensure that a variable contains a given string and that it's substituted.
#
define([UC_ENSURE], [dnl
ifelse($2, , [dnl
  $1=${$1-}
], [dnl
  for arg in $2; do
    case "$$1" in
      *$arg*[)] ;;
      *[)]      $1="$$1 $arg";;
    esac
  done
])dnl
AC_SUBST($1)dnl
]dnl
)


divert(diversion_number)dnl
