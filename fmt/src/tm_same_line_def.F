	LOGICAL FUNCTION TM_SAME_LINE_DEF ( line1, line2 )

* Check to see if two lines share an identical definition
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 2/92 *sh* - extracted from TM_FIND_LIKE_LINE by *mv*
* netCDF support *sh* 6/92 - weakened precision of line_mem comparison
* *sh* 12/92 (for netCDF) - use TM_FPEQ to weaken start/delta check, too
*              (should probably use it in the irregular coords check, too??)
* IBM port 14apr94 *kob* : Logical expressions need to be compared
*			   using NEQV and EQV
*			 : syntax error with "val**-5", it must be
*			   "val**(-5)"
* *sh* 2/24/95 - instead of comparing the units string compare the units code
* *sh* 10/95 - slightly different check if a dynamic line

* Include files
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
#endif

* Argument definitions
	INTEGER*4	line1, line2

* Local definitions
        LOGICAL         TM_FPEQ, dynamic
	INTEGER*4	i2, i1, j

* local parameter declaration
* ... all these are somewhat arbitrary
	REAL*8	    haf_dp_eps, max_sp_compare
	PARAMETER ( max_sp_compare = 2.**24,
     .              haf_dp_eps = 10.D0**(-14) ) ! somewhat arbitrary
	REAL	    haf_sp_eps
	PARAMETER ( haf_sp_eps = 10.0**(-7) )	! somewhat arbitrary

* Same regularity/irregularity?
	  IF (line_regular(line1) .NEQV. line_regular(line2)) GOTO 1000

* Same modulo?
	  IF (line_modulo(line1) .NEQV. line_modulo(line2)) GOTO 1000

* Check length
	  IF (line_dim(line1) .NE. line_dim(line2)) GOTO 1000

* Check units (2/95 - either a recognized unit code or identical strings)
	  IF (line_unit_code(line1) .NE. line_unit_code(line2)) GOTO 1000
	  IF (line_unit_code(line1) .EQ. 0 ) THEN
	     IF (line_units(line1) .NE. line_units(line2)) GOTO 1000
	  ENDIF
* Check orientation
	  IF (line_direction(line1) .NE. line_direction(line2)) GOTO 1000

* Check special values for time axis
	  IF (line_direction(line2) .EQ. 'TI') THEN
	    IF (line_t0(line2) .NE. line_t0(line1)) GOTO 1000
	    IF (line_tunit(line2) .NE. line_tunit(line1)) GOTO 1000
	  ENDIF

* dynamic lines
	  dynamic = line1.GT.max_lines .OR. line2.GT.max_lines
	  IF ( dynamic ) THEN
	    IF ( line1.LE.max_lines .OR. line2.LE.max_lines ) GOTO 1000
	    IF ( line_class (line1) .NE. line_class (line2) ) GOTO 1000
	    IF ( line_parent(line1) .NE. line_parent(line2) ) GOTO 1000
	  ENDIF

* check coordinate and box locations
	  IF (line_regular(line2)) THEN
            IF ( .NOT. TM_FPEQ( SNGL(line_start(line1)),
     .                          SNGL(line_start(line2)) ) ) GOTO 1000
            IF ( .NOT. TM_FPEQ( SNGL(line_delta(line1)),
     .                          SNGL(line_delta(line2)) ) ) GOTO 1000
	  ELSEIF (.NOT.dynamic) THEN
* static irregular lines
	    i2 = line_subsc1(line2)
	    i1 = line_subsc1(line1)
	    DO 100 j = 0,2*line_dim(line1)
              IF ( ABS(line_mem(i2+j)) .LT. max_sp_compare ) THEN
* ... single precision comparison
	         IF ( SNGL(line_mem(i2+j))
     .           .EQ. SNGL(line_mem(i1+j)) ) GOTO 100
* ... they're not equal but they may be within acceptable limits
	         IF ( ABS((line_mem(i2+j)-line_mem(i1+j))
     .		      /(line_mem(i2+j)+line_mem(i1+j))) .GT. haf_sp_eps )
     .							    GOTO 1000
              ELSE
* ... full precision comparison (needed for very large time words)
	         IF ( line_mem(i2+j) .EQ. line_mem(i1+j) ) GOTO 100
* ... they're not equal but they may be within machine accuracy
	         IF ( ABS((line_mem(i2+j)-line_mem(i1+j))
     .		      /(line_mem(i2+j)+line_mem(i1+j))) .GT. haf_dp_eps )
     .							    GOTO 1000
              ENDIF
  100	    CONTINUE
	  ENDIF

* the lines are identical
	TM_SAME_LINE_DEF = .TRUE.
        RETURN

* the lines differ
 1000   TM_SAME_LINE_DEF = .FALSE.
        RETURN
	END
