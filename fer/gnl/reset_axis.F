      SUBROUTINE RESET_AXIS (dset, varid, varname, attname, attbuff, 
     .     val, status) 


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
* execute the SET command

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA 
* 3/20/2006
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 
*
* Set axis attributes as in SET AXIS; called by edit_attribute.F
* doing SET ATT axname.attribute=

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'tmap_errors.parm'
#	include "tmap_dset.parm"
        include 'xtm_grid.cmn_text'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xunits.cmn_text'
	include 'xvariables.cmn'
        include 'calendar.decl'
        include 'calendar.cmn'

* Argument declarations

      CHARACTER*(*) varname, attname, attbuff
      INTEGER dset, varid, status
      REAL val

* Internal declarations

      REAL*8  TM_WW_AXLEN, axwwlen
      INTEGER STR_SAME, TM_GET_LINENUM, TM_LENSTR1, TM_UNIT_ID, 
     .        TM_GET_CALENDAR_ID, axis, i, since_T0, units, cal_id, 
     .        year, month, day, hour, minute, second, slen
      LOGICAL TM_HAS_STRING, its_changed, ok_cal
      CHARACTER*128 buff1

        buff1 = attname
 	axis = TM_GET_LINENUM( varname)
        IF ( axis .EQ. unspecified_int4 ) GOTO 5100
* MODULO
	IF ( STR_SAME(buff1, 'modulo') .EQ. 0) THEN
	   IF (val .EQ. unspecified_val4) THEN
	      val = 0.0
	   ELSE
	      val = ABS(val)  ! ABS to tolerate negatives
	   ENDIF
	   its_changed = .NOT.line_modulo(axis)
     .		    .OR.  val .NE. line_modulo_len(axis)
           line_modulo( axis ) = .TRUE.
	   line_modulo_len( axis ) = val

* ... wipe memory clear of stored variables - this could change all definitions
*      ( could be more selective for efficiency by checking to see where this
*      axis gets used )
	   IF ( its_changed ) THEN
              DO i = 1,max_mr_avail
                 IF ( mr_protected( i ) .NE. mr_deleted ) 
     .                                CALL DELETE_VARIABLE( i )
     	      ENDDO
           ENDIF

           RETURN
        ENDIF

* DEPTH
	IF ( STR_SAME(buff1, 'depth') .EQ. 0) THEN
           IF ( line_direction(axis) .EQ. 'DU' ) THEN
	      line_direction(axis) = 'UD'
           ENDIF

           RETURN
        ENDIF


* POSITIVE
	IF ( STR_SAME(buff1, 'positive') .EQ. 0) THEN
           IF ( STR_SAME(attbuff,'up') .EQ. 0) THEN
	      line_direction(axis) = 'DU'
           ELSE IF ( STR_SAME(attbuff,'down') .EQ. 0) THEN
	      line_direction(axis) = 'UD'
           ELSE
              GOTO 5200
           ENDIF

           RETURN
        ENDIF

* CALENDAR 

        IF (STR_SAME(buff1, 'calendar') .EQ. 0) THEN
           IF ( line_direction(axis) .EQ. 'TI' ) THEN

              ok_cal = .FALSE.

              IF (attbuff(1:3) .EQ. '360') attbuff = '360_DAY' 
              IF (attbuff(1:3) .EQ. 'STA') attbuff = 'GREGORIAN' 
              IF (attbuff(1:3) .EQ. 'PRO') attbuff = 'GREGORIAN' 
              IF (attbuff(1:3) .EQ. 'COM') attbuff = 'NOLEAP' 
              IF (attbuff(1:3) .EQ. '365') attbuff = 'NOLEAP' 
              IF (attbuff(1:3) .EQ. '366') attbuff = 'ALL_LEAP' 

              DO 200 i = 1, mxcals
                 IF (attbuff(1:3) .EQ. allowed_calendars(i)(1:3) ) THEN
                    ok_cal = .TRUE.
                    line_cal_name(axis) = allowed_calendars(i)
                 ENDIF
200          CONTINUE

              IF (.NOT. ok_cal) GOTO 5300

           ENDIF

           RETURN
        ENDIF

* TIME 0RIGIN 

        IF (STR_SAME(buff1, 'time_origin') .EQ. 0) THEN
           IF ( line_direction(axis) .EQ. 'TI' .OR.
     .          line_direction(axis) .EQ. 'TT' ) THEN

              CALL TM_BREAK_DATE (attbuff, 1, year, month, day,
     .				  hour, minute, second, status)
              IF ( status .NE. ferr_ok ) GOTO 5400
              line_t0(axis) = attbuff
              line_direction(axis) = 'TI'

           ENDIF

           RETURN
        ENDIF

*  UNITS
* Note we do not change line direction with a resetting of units; line
* direction UD, DU, WE, SN stay as they were when they were input.

        IF (STR_SAME(buff1, 'units') .EQ. 0) THEN

           since_T0 = MAX( INDEX(attbuff,'since'), 
     .        INDEX(attbuff,'SINCE') )
           IF (since_T0 .GT. 0) GOTO 5500

           line_units(axis) = attbuff

* ... decode the units (code lifted from cd_get_1_axis)

           units = TM_UNIT_ID( line_units(axis) )

* Named calendar with unit=year gets length of year in that calendar.

           cal_id = TM_GET_CALENDAR_ID (line_cal_name(axis))
           IF (units .EQ. -6 .OR. units .EQ. -10) THEN
              IF (cal_id .EQ. d360) units = -9
              IF (cal_id .EQ. julian) units = -12
              IF (cal_id .EQ. noleap) units = -11
              IF (cal_id .EQ. all_leap) units = -8
           ENDIF

           line_unit_code(axis) = units
           IF  ( units.EQ.0 .AND. line_units(axis) .NE. ' ' ) THEN
              CALL WARN ('Units  not recognized: '//line_units(axis))
              CALL WARN ('They will not be convertible:')
           ENDIF
           IF ( units .GE. 0 ) THEN
              line_tunit(axis) = real4_init
           ELSE
              line_tunit(axis) = un_convert(line_unit_code(axis))
           ENDIF

* See if we are resetting from a generic XX or YY axis to a geographic one.

           IF (line_direction(axis) .EQ. 'XX') THEN
              IF ( units .EQ. 4 .AND. 
     .              TM_HAS_STRING(line_units(axis), 'deg')
     .             .OR. TM_HAS_STRING(line_units(axis), 'lon') ) THEN 
                 line_direction(axis) = 'WE'

* Should we set axis to modulo?
                 axwwlen = TM_WW_AXLEN(axis)
                 IF ( axwwlen .LE. 360.D0 ) THEN
                    line_modulo(axis) = .TRUE.
                    IF (line_modulo_len(axis) .EQ. 0.0D0)
     .                    line_modulo_len(axis) = 360.D0
                 ENDIF

              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'YY') THEN
              IF ( units .EQ. 4 .AND. 
     .                  TM_HAS_STRING(line_units(axis), 'deg')
     .             .OR. TM_HAS_STRING(line_units(axis), 'lat') ) THEN 
                    line_direction(axis) = 'SN'
              ENDIF
           ENDIF

* recheck units and see if we are resetting from a geographic axis to a 
* non-geographic one or a calendar to non-calendar.

           IF (line_direction(axis) .EQ. 'WE' .OR. 
     .         line_direction(axis) .EQ. 'XX' ) THEN
              IF ( units .NE. 4 ) THEN 
                 line_direction(axis) = 'XX'
                 IF (units .LT. 0) THEN
                    CALL WARN ('Units  not recognized for X axis: '
     .                 //line_units(axis))
                    CALL WARN ('They will not be convertible')
                 ENDIF
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'SN' .OR. 
     .         line_direction(axis) .EQ. 'YY' ) THEN
              IF ( units .NE. 4 ) THEN 
                 line_direction(axis) = 'YY'
                 IF (units .LT. 0) THEN
                    CALL WARN ('Units  not recognized for Y axis: '
     .                 //line_units(axis))
                    CALL WARN ('They will not be convertible')
                 ENDIF
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'DU' .OR. 
     .         line_direction(axis) .EQ. 'UD' ) THEN
               IF (units .LT. 0) THEN
                 CALL WARN ('Units  not recognized for Z axis: '
     .                 //line_units(axis))
                 CALL WARN ('They will not be convertible')
              ENDIF
           ENDIF

           IF (line_direction(axis) .EQ. 'TI' .OR. 
     .         line_direction(axis) .EQ. 'TT' ) THEN
              IF ( units .GE. 0 ) THEN 
                 line_direction(axis) = 'TT'
                 CALL WARN ('Units  not recognized for T axis: '
     .                 //line_units(axis))
                 CALL WARN ('They will not be convertible')
              ENDIF
           ENDIF


        ENDIF

	RETURN

* error exits

 5000	RETURN

 5100 buff1 = varname
      slen = TM_LENSTR1(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .		 'unknown axis: '//buff1(1:slen),
     .               *5000 )
 5200 CALL ERRMSG( ferr_syntax, status,
     . 	         'positve may take the value UP or DOWN ',
     .               *5000 )
      
 5300 buff1 = attbuff
      slen = TM_LENSTR1(buff1)
      CALL ERRMSG( ferr_invalid_command, status,
     .	         'invalid calendar name: '
     .           //buff1(1:slen), 
     .               *5000 )

 5400 buff1 = attbuff
      slen = TM_LENSTR1(buff1)	
      CALL ERRMSG( ferr_syntax, status,
     .	         'invalid date string: '
     .           //buff1(1:slen), 
     .               *5000 )

 5500 CALL ERRMSG( ferr_syntax, status,
     .	         'specify TIME ORIGIN separately from UNITS ',
     .               *5000 )
      END
