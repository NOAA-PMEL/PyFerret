	INTEGER FUNCTION TM_GET_VARNUM( vname, dset )

* given a variable name supposed to be in the indicated data set return the
* TMAP variable ID

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/92

* calling argument declarations
	INTEGER		dset
	CHARACTER*(*)	vname

* include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

* local variable declarations:
      INTEGER STR_UPCASE, TM_LENSTR1, istat, slen, ivar
      CHARACTER upcase_var*24

* convert to upper case
      istat = STR_UPCASE( upcase_var, vname )     ! 4/91
      slen  = TM_LENSTR1( vname )

* locate the variable code among the variables in this data set
      DO 100 ivar = 1,maxvars

          IF ( ds_var_setnum(ivar) .NE. dset ) GOTO 100

          IF ( ds_var_code(ivar)(:slen) .EQ. upcase_var(:slen) ) THEN
            TM_GET_VARNUM = ivar
            RETURN
          ENDIF

 100  CONTINUE

* not found
      TM_GET_VARNUM = unspecified_int4
      RETURN
      END
