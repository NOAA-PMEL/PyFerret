	LOGICAL FUNCTION TM_SAME_LINE_DEF ( line1, line2 )

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* V510 4/00 *sh* - netCDF data sets using dynamic axes and grids
* V530 10/00 *sh* - bug fix: since netCDF axes are now dynamic, the decision
*		    to bypass coordinate checking on irregular axes must be
*		    based on these be child axes, rather than being dynamic
*		  - also switched to using TM_WORLD rather than direct refs
*		    to line_mem
*      3/01 *sh* - bug fix to 10/00 bug fix ... Still need to make sure
*	            it is the *same* child of an irreg axis by checking
*	            the lo/hi/delta
* V541 2/02 *sh* - added support for line_modulo_len

* V570 5/04 *acm* - Use TM_DFPEQ to weaken  irregular coords check
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
* V614 *acm* 10/08 - Fix bug 1595, comparisons of xx1 and -1*xx2 did
*                   not take into account that the values may be zero.
* V720 *sh* 4/17 - special treatment for DSG axes

* Include files
	include 'tmap_dims.parm'
	include 'implicit.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data

* Argument definitions
	INTEGER*4	line1, line2

* Local definitions
        LOGICAL         TM_FPEQ, TM_FPEQ_SNGL, TM_DFPEQ, dynamic
	INTEGER*4	TM_LENSTR1, STR_CASE_BLIND_COMPARE, len, istat, j
	REAL*8		TM_WORLD_AX, xx1, xx2
	REAL*4  asingle, bsingle

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

* Same modulo length?
	  IF (line_modulo(line1)) THEN
	     IF (line_modulo_len(line1) .NE. line_modulo_len(line2))
     .								GOTO 1000
	  ENDIF

* Check length
	  IF (line_dim(line1) .NE. line_dim(line2)) GOTO 1000

* Check units (2/95 - either a recognized unit code or identical strings)
	  IF (line_unit_code(line1) .NE. line_unit_code(line2)) GOTO 1000
	  IF (line_unit_code(line1) .EQ. 0 ) THEN
	     IF (line_units(line1) .NE. line_units(line2)) GOTO 1000
	  ENDIF
* Check orientation
	  IF (line_direction(line1) .NE. line_direction(line2)) GOTO 1000

* DSG axes are unique per dataset
	  IF (line_dsg_dset(line1) .NE. line_dsg_dset(line2) .AND. 
     .          line_dsg_dset(line1).NE.line_0 .AND. line_dsg_dset(line2).NE.line_0)   GOTO 1000

* Check special values for time axis
	  IF (line_direction(line2) .EQ. 'TI') THEN

	    len = TM_LENSTR1( line_t0(line1) )
	    istat = STR_CASE_BLIND_COMPARE(line_t0(line2)(:len),line_t0(line1)(:len))
	    IF (istat .NE. str_match) GOTO 1000
c	    IF (line_t0(line2) .NE. line_t0(line1)) GOTO 1000

	    IF (line_shift_origin(line2) .NEQV. line_shift_origin(line1))
     .           GOTO 1000
	    IF (line_tunit(line2) .NE. line_tunit(line1)) GOTO 1000
	    IF (line_cal_name(line1) .NE. line_cal_name(line2)) GOTO 1000
	  ENDIF

* dynamic line info
	  dynamic = line1.GT.max_lines .OR. line2.GT.max_lines
	  IF ( line_class (line1) .NE. line_class (line2) ) GOTO 1000
	  IF ( line_parent(line1) .NE. line_parent(line2) ) GOTO 1000

* check coordinate and box locations
	  IF ( line_regular(line2)
     .	.OR.   (line_class (line1) .EQ. pline_class_stride)
     .	.OR.   (line_class (line1) .EQ. pline_class_midpt)   ) THEN
	    asingle = SNGL(line_start(line1))
	    bsingle = SNGL(line_start(line2))
            IF ( .NOT. TM_FPEQ_SNGL( asingle, bsingle ) ) GOTO 1000
	    asingle = SNGL(line_delta(line1))
	    bsingle = SNGL(line_delta(line2))
            IF ( .NOT. TM_FPEQ_SNGL( asingle, bsingle ) ) GOTO 1000
	  ELSE
* irregular lines
	    DO 100 j = 1,line_dim(line1)
* first check the grid points
	      xx1 = TM_WORLD_AX(j, line1, box_middle)
	      xx2 = TM_WORLD_AX(j, line2, box_middle)
	      IF (.NOT. TM_DFPEQ(ABS(xx1), 0.D0) ) THEN
	         IF ( TM_DFPEQ(ABS(xx1), -1*ABS(xx2)) ) GOTO 1000
              ENDIF
              IF ( ABS(xx2) .LT. max_sp_compare ) THEN
* ... single precision comparison
	         IF ( SNGL(xx2) .EQ. SNGL(xx1) ) GOTO 100
* ... they're not equal but they may be within acceptable limits
	         IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_sp_eps )
     .							    GOTO 1000
              ELSE
* ... full precision comparison (needed for very large time words)
	         IF ( xx1 .EQ. xx2 ) GOTO 100
* ... they're not equal but they may be within machine accuracy
	         IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_dp_eps )
     .							    GOTO 1000
              ENDIF

* now the grid cell bounds
	      xx1 = TM_WORLD_AX(j, line1, box_lo_lim)
	      xx2 = TM_WORLD_AX(j, line2, box_lo_lim)
	      IF (.NOT. TM_DFPEQ(ABS(xx1), 0.D0) )  THEN
	         IF ( TM_DFPEQ(ABS(xx1), -1*ABS(xx2)) ) GOTO 1000
              ENDIF
              IF ( ABS(xx2) .LT. max_sp_compare ) THEN
* ... single precision comparison
	         IF ( SNGL(xx2) .EQ. SNGL(xx1) ) GOTO 100
* ... they re not equal but they may be within acceptable limits
	         IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_sp_eps )
     .							    GOTO 1000
              ELSE
* ... full precision comparison (needed for very large time words)
	         IF ( xx1 .EQ. xx2 ) GOTO 100
* ... they're not equal but they may be within machine accuracy
	         IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_dp_eps )
     .							    GOTO 1000
              ENDIF
  100	    CONTINUE

* finally the upper limit of the top cell
	    xx1 = TM_WORLD_AX(j, line1, box_hi_lim)
	    xx2 = TM_WORLD_AX(j, line2, box_hi_lim)
	      IF (.NOT. TM_DFPEQ(ABS(xx1), 0.D0) )  THEN
	         IF ( TM_DFPEQ(ABS(xx1), -1*ABS(xx2)) ) GOTO 1000
              ENDIF
            IF ( ABS(xx2) .LT. max_sp_compare ) THEN
* ... single precision comparison
	       IF ( SNGL(xx2) .EQ. SNGL(xx1) ) GOTO 500
* ... they're not equal but they may be within acceptable limits
	       IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_sp_eps )
     .							    GOTO 1000
              ELSE
* ... full precision comparison (needed for very large time words)
	       IF ( xx1 .EQ. xx2 ) GOTO 500
* ... they're not equal but they may be within machine accuracy
	       IF ( ABS((xx2-xx1)/(xx2+xx1)) .GT. haf_dp_eps )
     .							    GOTO 1000
            ENDIF

	  ENDIF

* the lines are identical
 500	  TM_SAME_LINE_DEF = .TRUE.
        RETURN

* the lines differ
 1000   TM_SAME_LINE_DEF = .FALSE.
        RETURN
	END
