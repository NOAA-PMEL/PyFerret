	SUBROUTINE ASCII_LIMITS ( cx, idim, string, slen )

*
*
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
* produce a formatted string representing the space or time limits along the
* specified axes within the specified context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/1/86
* revision 0.1 - 1/9/87  - added trans_integrated
* revision 0.2 - 4/28/87 - automated transformation comment using ALG_TRANS_TITL
* revision 0.3 - 7/13/87 - removed axis_titles - squeezed out unnedded blanks
* revision 0.4 - 9/10/87 - added "using nnn pts" for transformations w/ args
* revision 0.5 - 9/26/87 - using routine LOHI_WORLD
* revision 0.6 - 10/23/87- "using"-->"by" and nnn reduced to minimum digits
* revision 0.7 - 3/25/88 - increased LOHI_WORLD size for date/time strings
* V200:  1/12/90 - "of nnn" text string non-integer transform args
* V430:  7/96 - pass cx to ALG_TRANS_TITL so it can document 4D transformations
* v530  *acm* 3/01 - Show calendar name for alternative calendar types.

#ifdef unix
	include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xalgebra.cmn'
	include 'calendar.decl'
	include 'calendar.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, idim, slen
	CHARACTER*(*)	string

* local parameter declarations:
	LOGICAL		full_precision
	PARAMETER     ( full_precision = .TRUE. )

* internal variable declarations:
	INTEGER		TM_LENSTR, TM_GET_CALENDAR_ID, 
     .			max_len, len_trans, llen, trans, i,
     .			grid, cal_id, len_cal
	CHARACTER	ALG_TRANS_TITL*20, LOHI_WORLD*44, LEFINT*4, 
     .			TM_FMT*10, TM_GET_CAL*32,
     .			trans_title*40, cal_name*32
	REAL		trans_arg

* initialize
	max_len	= LEN(string)
	trans     = cx_trans( idim, cx )
	trans_arg = cx_trans_arg( idim, cx )

* not an applicable dimension ?
	IF ( cx_lo_ww( idim, cx ) .EQ. unspecified_val8 ) THEN
	   string     = ' N/A'
	   slen = 4
	   RETURN
	ENDIF

* put in lower/upper world coordinates
	string = LOHI_WORLD( idim, cx, ' to ', full_precision, slen )
	
* transformation requiring comment ?
	IF ( trans .NE. trans_no_transform ) THEN
	   trans_title = ALG_TRANS_TITL( trans, cx )
	   len_trans   = TM_LENSTR( trans_title )
	   IF ( trans_arg .NE. bad_val4 ) THEN
	      DO 100 i = 1, num_transforms
 100	      IF ( trans .EQ. alg_trans_num(i) ) GOTO 200
	      STOP 'ASCII_LIMITS'
 200	      IF ( alg_trans_its_int(i) ) THEN
* ... add "by nnn pts" squeezing "nnn" to minimum digits
	         trans_title = trans_title(:len_trans)//' by '//
     .			       LEFINT( INT(trans_arg), llen )
	         len_trans = len_trans + 4 + llen
	         trans_title = trans_title(1:len_trans)//' pts'
	         len_trans = len_trans + 4
	      ELSE
* ... add "of nnn " in a nice format
	         trans_title = trans_title(:len_trans)//' of '//
     .			       TM_FMT( trans_arg, 4, 10, llen )
	         len_trans = len_trans + 4 + llen
	      ENDIF
	   ENDIF
	   string = string(1:slen)//' ('//trans_title(:len_trans)//')'	
	   slen = MIN( max_len, slen+len_trans+3 )
	ENDIF

* calendar name requiring comment ?

	IF (idim .EQ. t_dim  .AND. 
     .      cx_lo_ww( idim, cx ) .NE. unspecified_val8 ) THEN

	   grid = cx_grid(cx)
	   cal_name = TM_GET_CAL (grid)
	   cal_id = TM_GET_CALENDAR_ID (cal_name)
	   IF (cal_id .GT. gregorian .AND. cal_id .LT. max_calendars) THEN
	      len_cal = TM_LENSTR(cal_name)
	      string =  string(1:slen)//' '//cal_name(:len_cal)
	      slen = MIN( max_len, slen+len_cal+1 )
	   ENDIF
	ENDIF

	RETURN
	END
