	SUBROUTINE TM_INTERNAL_READ(fstring, value, status)
*Will take a fortran character array and extract the floating point value
* it contains.  Needed for IBM-only internal reads
*
*       -kob- 10/95
*
* ARGUMENT DEFINITIONS
* fstring  - fortan string holding desired fp value
* date  - date to process
* status - returned status
	CHARACTER*(*)	fstring
	REAL		value
	INTEGER	 	status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
#endif

* internal variable declarations:
	INTEGER slen
	PARAMETER ( slen = 32)
	INTEGER TM_LENSTR1, TM_INTERNAL_READ_C, dlen

#ifdef sun
	BYTE       dhol(slen)     !c-type Hollerith string buffers
#else
	INTEGER*1  dhol(slen)     !c-type Hollerith string buffers
#endif

	dlen = TM_LENSTR1( fstring)

	CALL TM_FTOC_STRNG(fstring(:dlen), dhol, slen)
	status = TM_INTERNAL_READ_C(dhol, value)


* successful translation
	status = merr_ok
	RETURN

* error exits
 5000	RETURN

	END
