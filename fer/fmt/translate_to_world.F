	SUBROUTINE TRANSLATE_TO_WORLD ( val, idim, grid, ndec, string )

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
* convert a coordinate value to a string using appropriate lat/long/depth/time
* format.  String will be left justified within the given field width.  Number
* of decimal places of precision is specified by the argument ndec

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 7/17/86
* revision 0.1 - 7/14/87 - allowed field widths of 3 digits
* revision 1.0 - 2/25/88 - added option for non-fancy coordinates
* revision 2.0 -  4/1/88 - incorporated time/date strings and FIELD_WIDTH
*			   and ndec<0 indicating to use significant
*			   digits instead of decimal places
* revision 2.1 - 4/13/88 - added MOD calculation to longitude
* revision 2.2 - 1/12/89 - declate TM_SECS_TO_DATE exactly *20
* revision 2.3 - 2/16/89 - TM_SECS_TO_DATE --> SECS_TO_DATE
* V200:  1/10/89 - kludge to restrict degrees to precision .001 avoid ".000001N"
* Unix/RISC port 11/91  *sh*: check for unspecified_val8 value, exit after '*'
* V230:  7/20/92 - check T0 before assuming that times can be formatted
*        7/23/92 - for modulo longitude calculations include numerical value
*                  in parentheses
* V300:   2/3/93 - bug fix negative times (dates) - should be always "fancy"
* V320: 12/20/94 - do not label "m" on Z axis -- in order to support the
*		   meteorology community with millibars, etc.
*	 1/27/95 - correct the longitude formatting of negative longitudes
* V530: *acm* 12/00 alternative calendar types; cal_id
* V533: *sh* 7/01 - bug fix: negative time steps still formatted as seconds
*		    since BC.  A leftover from an Oct. 1997 change by K.O.
*	            omitted from this documentation block
* V580 *acm* 10/01  modify the kludge to restrict degrees to precision 0.001
*                   If the calling sequence asks for more precision, allow it.
* V581 *acm*  3/05  axis-independent time coordinates are calculated with default 
*                   calendar, so do not get cal_id when converting to date string
*       *acm* 4/12 6D Ferret: time axis may be in t or f direction. 
* v75  *acm* 3/18  Issue 1856, changes to allow for timesteps in fractional seconds
* v74+ *acm* 11/18  Issue 1902, fractional seconds only when time units are seconds
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'errmsg.parm'
	include 'ferret.parm'

* calling argument declarations:
	INTEGER		idim, ndec, grid, line
	CHARACTER*(*)	string
	REAL*8		val

* internal variable declarations:
	CHARACTER	SECS_TO_DATE*20, TM_FMT*64, TM_SECS_TO_DATE*20, 
     .                  code*1, xtra*16, buff*20
	INTEGER		FIELD_WIDTH, TM_GET_CALENDAR_ID, TM_LENSTR1, 
     .                  min_width, width, decim, slen, n2, cal_id, iv, status
	LOGICAL		GEOG_LABEL, fancy, use_sig_fig,
     .                  modulo, truncate, no_year
	REAL		TSTEP_TO_SECS, val4, val4_orig, fval

* valid inputs ?  (bug fix 11/91)
        IF ( val .EQ. unspecified_val8 ) THEN
           string = '*'
           RETURN
        ENDIF

* initialize
        modulo     = .FALSE.
	width	   = MIN( 30, LEN(string) )
	use_sig_fig = ndec .LT. 0			! use TM_FMT
	decim	   = ABS(ndec)
	val4	   = val
        val4_orig  = val4
	code	   = ' '
        fancy      = GEOG_LABEL( idim, grid )
	truncate   = .FALSE.

! Note (2/3/93): GEOG_LABEL contains the removed logic

* adjust precision to fit into field if necessary
100	min_width = FIELD_WIDTH( val, grid, idim, decim, slen )
	IF ( width .LT. min_width ) THEN
	   IF ( decim .EQ. 0 ) THEN
	      string = '*'		! too narrow to format
              RETURN                    ! bug fix 11/91
	   ELSEIF ( decim .GT. ABS(ndec)) THEN
	      truncate = .TRUE.
	      decim = ABS(ndec)
	      min_width = width
	   ELSE
	      decim = decim - 1
	      GOTO 100
	   ENDIF
	ENDIF

* is it a fancy (lat/long/depth/time) position ?
	IF ( fancy ) THEN

	   IF ( idim .EQ. x_dim ) THEN
* x - longitude E/W
* ... restrict degrees to a precision of .001
c (The following doesnt always compute the correct value in double precision
c on 32-bit linux (.145 comes back as .146). Use NINT instead.)
c	      IF (ndec .LE. 3) val4 = INT(val4 * 1000.D0)/1000.D0
	      IF (ndec .LE. 3) val4 = NINT(val4 * 1000.D0)/1000.D0
	      val4 = MOD( val4, 360. )
	      IF ( val4 .LT. 0.0 ) val4 = val4 + 360.	! 1/27
              modulo = val4 .NE. val4_orig
	      IF     ( val4 .LE. 180.0 ) THEN
	         code    = 'E'			! degE
	      ELSEIF ( val4 .GT. 180.0 ) THEN
	         val4 = 360. - val4
	         code    = 'W'			! degW
	      ENDIF

	   ELSEIF ( idim .EQ. y_dim ) THEN
* y - latitude N/S
* ... restrict degrees to a precision of .001
c (The following doesnt always compute the correct value in double precision
c on 32-bit linux (.145 comes back as .146). Use NINT instead.)
c	      IF (ndec .LE. 3) val4 = INT(val4 * 1000.D0)/1000.D0
	      IF (ndec .LE. 3) val4 = NINT(val4 * 1000.D0)/1000.D0
 	      IF     ( val4 .LT. 0.0 ) THEN
	         val4 = -val4
	         code    = 'S'			! degS
	      ELSEIF ( val4 .GT. 0.0 ) THEN
	         code    = 'N'			! degN
	      ENDIF

	   ELSEIF ( idim .EQ. z_dim ) THEN
* z - depth in meters
!	      code = 'm'				! meters
	      code = ' '		! 12/94 change

	   ELSE
* t - create date/time string and exit
*  ( negative times are used in stored regions as axis-independent dates )
              IF ( val .LT. -47335449600D0 ) THEN
! HACK! negatives time steps are formatted as valid times unless they
! would be negative seconds from BC for dates later than 1-jan-1500
                 line       = grid_line( idim, grid )
!                 cal_id = TM_GET_CALENDAR_ID ( line_cal_name(line))
                 cal_id = 1  ! axis-independent dates were calculated with dflt calendar
                 string = SECS_TO_DATE( -val, cal_id )
              ELSE
                 CALL TSTEP_TO_DATE( grid, idim, val, 6, string )
		 no_year = (TM_LENSTR1(string) .LT. 19)
                 CALL TSTEP_TO_DATE( grid, idim, val, decim, string )
                 slen = TM_LENSTR1(string)
                 line       = grid_line( idim, grid )
		 cal_id = TM_GET_CALENDAR_ID ( line_cal_name(line))

		 buff = string
		 CALL CHECK_DATE_REFORMAT (string, slen, cal_id, status)
* for ISO dates keep the 4-digit year
		 IF (buff .NE. string .AND. no_year) THEN  
		    fval = TSTEP_TO_SECS( grid, idim, val )
		    string = TM_SECS_TO_DATE( fval, cal_id )
		    CALL CHECK_DATE_REFORMAT (string, slen, cal_id, status)
		 ENDIF

		 line    = grid_line( idim, grid )
                 IF (decim.GT.7 .AND. line_tunit(line) .EQ. 1.) THEN  ! units of data in seconds
                    fval = AINT(val)
                    IF (fval .NE. val) THEN
                       fval = val-fval
                       xtra = TM_FMT(fval, 4, 16, n2)  
                       IF (width-slen .GE. n2-1) 
     .                    string = string(:slen)//xtra(2:n2)
                    ELSE 
		       string = string(:slen)//'.0'
		    ENDIF
                 ENDIF
                 
		 IF (truncate) THEN
		    string(slen-3:slen) = '***'
		 ENDIF

              ENDIF
	      RETURN
	   ENDIF
	ENDIF

* create all strings except date/time here
	IF ( use_sig_fig ) THEN
	   string = TM_FMT( val4, decim, width, slen )
	ELSE
	   string = TM_FMT( val4, -decim, width, slen )
	ENDIF
	string = string(:slen)//code

* add parenthetical remark if modulo translation has occurred
        IF ( modulo ) THEN
	   IF ( use_sig_fig ) THEN
              xtra =  TM_FMT( val4_orig, decim, width, n2 )
	   ELSE
              xtra =  TM_FMT( val4_orig, -decim, width, n2 )
	   ENDIF
           IF ( LEN(string) .GE. slen+n2+3 ) 
     .                string = string(:slen+1)//'('//xtra(:n2)//')'
        ENDIF


	RETURN
	END
