	SUBROUTINE SHOW_LINE( line, brief )

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
* display basic facts about the given line

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88
* V200:  9/22/89 - added "orient" for axes without particular orientations
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V530:  3/01    - indicate the calendar for non-Gregorian calendar types 
* V541 *sh*  2/02 - describe axis span and modulo length
* V553 *acm* 6/03 - Allow for line_direction code of 'XX' or 'YY'
* V581 *acm* 6/05 - For fix to bug 1271, add flag line_shift_origin
*            and restore original t0 on output
* V650  Write to show_lun which may be set to a file unit number rather than std out
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
* V695+  *sh* 5/15 show T0 for forecast calendar axes, too
* V697  12/15 *acm* Changes to add details on SHOW GRID uvar to indicate 
*                   compressed or subsetted axes
* V720 4/17 *sh* - support for DSG contiguous ragged arrays

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'implicit.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include	'ferret.parm'
	include 'xprog_state.cmn'
	include 'xtext_info.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'

* calling argument declarations
	LOGICAL		brief
	INTEGER 	line

* local variable declarations:
	LOGICAL		TM_ITS_SUBSPAN_MODULO
	INTEGER		idim, orient, slen, i, idum
	INTEGER		TM_LENSTR1, TM_GET_CALENDAR_ID, cal_id
	CHARACTER	TM_FMT*48, buff*80, buff20*20
	REAL*8		TM_WW_AXLEN

* determine orientation
	DO 100 idim = 1, nferdims+1
	   IF ( line_direction(line) .EQ. axis_orients( idim ) ) THEN
	      orient = idim
	      GOTO 110
	   ENDIF
 100	CONTINUE
        IF (line_direction(line) .EQ. 'XX') THEN
           orient = 1
           idim = 1
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'YY') THEN
           orient = 2
           idim = 2
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'TT') THEN
           orient = 4
           idim = 4
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'EE') THEN
           orient = 5
           idim = 5
           GOTO 110
        ENDIF
        IF (line_direction(line) .EQ. 'FF') THEN
           orient = 6
           idim = 6
           GOTO 110
        ENDIF

	orient = 0
	idim   = 1				! use X slot for non-oriented
 110	IF ( idim .EQ. nferdims+1 ) idim = 3	! DU --> UD on Z axis

* create a dummy grid with this axis
* (kludge to use existing code that works with grids)
	grid_line(idim, mgrid_buff) = line

* report basic facts using code from SHOW_GRID
* Last argument is user-var code, not needed here, but used when calling from SHOW_GRID
	buff20 = ' '
	idum = -1
	CALL LINE_FACTS( show_lun, idim, mgrid_buff, orient, idum )
	IF ( brief ) RETURN
* nothing more to show if the "line" represents a contiguous ragged array
	IF (idim.NE.e_dim .AND. line_dsg_dset(line).NE.pline_not_dsg) RETURN
!DSGTBD Is there more we'd like to tell about the contiguous ragged array?

* report /T0= for formatted time axis
	IF  ( line_direction(line) .EQ. 'TI' 
     .	.OR.  line_direction(line) .EQ. 'FI' ) THEN
            IF (line_shift_origin(line)) THEN
               buff20 = '01-JAN-0001 00:00:00'
            ELSE
 	       buff20 = line_t0(line)
            ENDIF
	   CALL SPLIT_LIST
     .		(pttmode_explct, show_lun, 'T0 = '//buff20, 0)

* report the calendar of a time axis if "interesting"
	   buff = line_cal_name (line)
	   cal_id = TM_GET_CALENDAR_ID (buff)
	   IF (cal_id .GT. gregorian .AND. cal_id .LE. max_calendars) THEN
	      slen = TM_LENSTR1(buff)
	      CALL SPLIT_LIST (pttmode_explct, show_lun, 
     .		               'CALENDAR = '//buff(:slen), 0)
	   ENDIF
	ENDIF

* report the full span of the axis
	buff =  '   Axis span (to cell edges) = '//
     .		TM_FMT( TM_WW_AXLEN(line),7,22,slen )
	slen = slen + 31

* if line is modulo, report its modulo cycle length
	IF ( line_modulo(line) ) THEN
	   IF ( TM_ITS_SUBSPAN_MODULO(line) ) THEN
	     buff = buff(:slen)// ' (modulo length = ' //
     .		   TM_FMT( line_modulo_len(line),7,22,i )
	     slen = slen + i + 18
	     buff = buff(:slen) //')'
	     slen = slen + 1
	   ELSE
	     buff = buff(:slen) // ' (modulo length = axis span)'
	     slen = slen + 28
	   ENDIF
	ENDIF
	CALL SPLIT_LIST (pttmode_explct, show_lun, buff, slen)

	RETURN
	END



