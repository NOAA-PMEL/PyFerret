	LOGICAL FUNCTION TM_Line_match ( line1, line2 )

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
*
* Check to see if line1 is identical to line2
*
* by M. Verschell PMEL/TMAP
*
* revision 0.00 - 10/19/88
* 5/5/93 - *sh* 2 bug fixes in coordinate compare of irregular axes
*	 - 1) reversed logic; 2) convert args to SNGL
*
* IBM port 14apr94 *kob* : Logical expressions need to be compared
*			   using NEQV and EQV
* 1/96 *sh* "bug" fix: match was based on exactly identical unit strings
*		this meant that "DEGREES" didn't match "LONGITUDE".
* V541 *sh* 2/02 - check line_modulo_len
* V551 *acm*2/03 - check calendar type
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin

* Include files
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'xtm_grid.cmn_text'
	external xgt_grid_data


* Argument definitions
	INTEGER*4	line1, line2

* Local definitions
	INTEGER*4	j, lc1, lc2
	LOGICAL		TM_FPEQ
	REAL		GET_LINE_COORD, v1, v2

* Same regularity/irregularity?
	IF (line_regular(line1) .NEQV. line_regular(line2)) GOTO 110

* Same modularity?
	IF (line_modulo(line1) .NEQV. line_modulo(line2)) GOTO 110

* Same modulo length?
	IF (line_modulo(line1)) THEN
	  IF (line_modulo_len(line1) .NE. line_modulo_len(line2))
     .							  GOTO 110
	ENDIF

* Check length
	IF (line_dim(line1) .NE. line_dim(line2)) GOTO 110

* Check units
	IF (line_unit_code(line1) .NE. line_unit_code(line2) ) GOTO 110
* ... check actual string for unrecognizable units
	IF ( line_unit_code(line1) .EQ. 0 ) THEN	
	   IF (line_units(line1) .NE. line_units(line2)) GOTO 110
	ENDIF

* Check orientation
	IF (line_direction(line1) .NE. line_direction(line2)) GOTO 110

* Check calendar type
	IF (line_direction(line1) .EQ. 'TI') THEN
           IF (line_cal_name(line1) .NE. line_cal_name(line2)) GOTO 110
        ENDIF

* Check special values for time axis
	IF (line_direction(line1) .EQ. 'TI') THEN
	  IF (line_t0(line1) .NE. line_t0(line2)) GOTO 110
	  IF (line_shift_origin(line1) .NEQV. line_shift_origin(line2)) 
     .        GOTO 110
	  IF (line_tunit(line1) .NE. line_tunit(line2)) GOTO 110
	ENDIF

* check coordinate and box locations
	IF (line_regular(line1)) THEN
	  IF (line_start(line1) .NE. line_start(line2)) GOTO 110
	  IF (line_delta(line1) .NE. line_delta(line2)) GOTO 110
	ELSE
	  lc1 = line_subsc1(line1)
	  lc2 = line_subsc1(line2)
	  DO 100 j = 1,line_dim(line1)
	    v1 = GET_LINE_COORD (linemem(line1)%ptr, j)
	    v2 = GET_LINE_COORD (linemem(line2)%ptr, j)
	    IF (.NOT.TM_FPEQ(v1, v2)) GOTO 110

	    v1 = GET_LINE_COORD (lineedg(line1)%ptr, j)
	    v2 = GET_LINE_COORD (lineedg(line2)%ptr, j)
	    IF (.NOT.TM_FPEQ(v1, v2)) GOTO 110
  100	  CONTINUE

* Upper box edges
	    v1 = GET_LINE_COORD (lineedg(line1)%ptr, line_dim(line1)+1)
	    v2 = GET_LINE_COORD (lineedg(line2)%ptr, line_dim(line1)+1)
	    IF (.NOT.TM_FPEQ(v1, v2)) GOTO 110

	ENDIF

* Yes, lines match
	TM_Line_match = .TRUE.
	GOTO 9999

* No, lines don't match
  110	TM_Line_match = .FALSE.

 9999	RETURN
	END
