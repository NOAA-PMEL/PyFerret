	LOGICAL FUNCTION TM_ABSTRACT_AXIS( iaxis )

* is this an axis that is so long it only the "used" portion of its
* length should be represented under some circumstances
* (an "abstract" axis in FERRET)

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 6/92
*   5/93 *sh* The axis EZ is also an "abstract" axis
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

* calling argument declarations:
        INTEGER iaxis

* internal variable declarations:

#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/xtm_grid.cmn_text'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

      IF ( iaxis .LT. 1 .OR. iaxis .GT. max_lines ) THEN
         TM_ABSTRACT_AXIS = .FALSE.
      ELSE
         TM_ABSTRACT_AXIS = line_name( iaxis ) .EQ. 'ABSTRACT'
     .	               .OR. line_name( iaxis ) .EQ. 'EZ'
      ENDIF

      RETURN
      END
