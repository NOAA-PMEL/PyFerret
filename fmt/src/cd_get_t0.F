        SUBROUTINE CD_GET_T0(filename, units_string, calendar_string,
     .                  agg_cal_id, unit_code, t0_string, status)
 
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*.
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
* Given the a CF time units string of the form "<time units> since <T0>"
* (or a historical bevvy of alternative formats from per cd_get_1_axis)
* extract the units and T0 string and convert T0 into a standard Vax format of
* dd-mon-yyyy:hh:mm:ss, where "mon" is a 3-letter month name

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V695+  10/15 *sh* new routine extracted from code in cd_get_1_axis
* V698  4/16 *acm* new TM_UNITS_CAL to account for calendar in units id

* argument definitions:
*       filename (input)
*               - name of file being processed
*       units_string (input)
*               - string associated with the units attribute in a CF file
*       calendar_string (input)
*               - string associated with the calendar attribute in a CF file
*       agg_cal_id (input) - id calendar used by file1 of the aggregation
*       unit_code(output)  - time units of this file
*       t0_string (output) - T0 string in standard Vax form
*       status    (output) - return status

* include files
        include 'tmap_errors.parm'
#include "gt_lib.parm"
        include 'calendar.decl'
        include 'errmsg.parm'
        include 'xio.cmn_text'

* argument declarations
        INTEGER  status, unit_code, agg_cal_id
        CHARACTER*(*) filename, units_string, calendar_string, t0_string

* local variable declarations:
        LOGICAL ok_cal, yyyymmdd_time, true_month
        INTEGER TM_LENSTR1, TM_UNIT_ID, TM_UNITS_CAL, 
     .           TM_GET_CALENDAR_ID, STR_UPCASE,
     .           flen, clen, slen, istat, since_T0
        CHARACTER cal_name*32, tunits*32, dbuf*3, buff*128
        REAL*8 TM_SECS_FROM_BC, 
     .         days_neg4713, days_1901_may68, secs_to_1901, secs_to_1968 

        INTEGER tt  ! nice short name
        EQUIVALENCE (tt,lunit_errors)

* parameter declarations
        REAL*8          psecs_per_day
        PARAMETER     ( psecs_per_day = 60.*60.*24. )

* initialize
        flen = TM_LENSTR1(filename)
        clen = TM_LENSTR1(calendar_string)
        t0_string = ' '

* check that the calendar of this file is consistent with the given agg_cal_id
* warning only if it is not
        IF (calendar_string .EQ. ' ') THEN
           cal_name= 'GREGORIAN'   ! default
        ELSE
           cal_name = calendar_string
        ENDIF
        CALL CD_GET_CALENDAR_NAME(cal_name, ok_cal)
        IF (ok_cal) ok_cal= agg_cal_id.EQ.TM_GET_CALENDAR_ID (cal_name)
        IF (.NOT.ok_cal) CALL TM_NOTE("calendar attribute "//
     .       calendar_string(:clen)//" in file "//filename(:flen)//
     .       " is inconsistent with file 1 of aggregation. Ignored", tt)

* units_string typ will contain "tunits since T0_date"
        buff = units_string
        since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
        yyyymmdd_time = buff(1:15) .EQ. 'yyyymmddhhmmss'

        IF ( since_T0 .GT. 2 ) THEN
           tunits = buff(:since_T0-1)  ! just the "units" name
           istat = STR_UPCASE( dbuf, buff(1:3) )
           slen = TM_LENSTR1(buff)
           IF ( slen .GT. since_T0 + 5) THEN
              IF (buff(since_T0+6:since_T0+10) .EQ. '-4713' .AND.
     .            dbuf .EQ. 'DAY')  THEN

c  This is the simple conversion; but we will reset the origin to 1901.
c               buff = 'DAYS since 1968-05-23 00:00:00'
c               days_neg4713 = 2440000.D0

                 buff = 'DAYS since 1901-01-15 00:00:00'
                 secs_to_1901 = TM_SECS_FROM_BC 
     .                        (GREGORIAN, 1901,1,15,0,0,0,status)
                 secs_to_1968 = TM_SECS_FROM_BC 
     .                        (GREGORIAN, 1968,5,23,0,0,0,status)
                 days_1901_may68 = (secs_to_1968 - secs_to_1901)/
     .                             psecs_per_day
                 days_neg4713 = 2440000.D0 - days_1901_may68
              ENDIF
              CALL TM_DATE_REFORMAT(buff(since_T0+6:),pdate_numslashnum,
     .                            agg_cal_id, t0_string,
     .                            pdate_vax, .FALSE., status )

c fix for bug 1394
c If the part after since is not a date/time, just keep the whole
c string as the units.  i.e., "days_since_event".
              IF ( status .NE. merr_ok ) THEN   
                 tunits = buff(:slen)
                 t0_string = ' '
                 status = merr_ok
              ENDIF
           ELSE
              T0_string = ' '        ! no T0 is present
           ENDIF
        ELSEIF ( yyyymmdd_time ) THEN
           tunits = 'days'
           t0_string  = 'yyyymmdd' ! relevant?  untested code as of 11/2015
        ELSE
           tunits = buff
           t0_string = ' '
        ENDIF

* decode the units
        unit_code = TM_UNIT_ID( tunits )

* Named calendar with unit=year gets length of year in that calendar.
      IF (unit_code .EQ. -6 .OR. unit_code .EQ. -10) THEN
         IF (agg_cal_id .EQ. d360) unit_code = -9
         IF (agg_cal_id .EQ. julian) unit_code = -12
	 IF (agg_cal_id .EQ. noleap) unit_code = -11
	 IF (agg_cal_id .EQ. all_leap) unit_code = -8
      ENDIF

* Named calendar with unit=month gets length of month in that calendar.
      IF (unit_code .EQ. -5) THEN
         IF (agg_cal_id .EQ. d360) unit_code = -14
         IF (agg_cal_id .EQ. julian) unit_code = -15
	 IF (agg_cal_id .EQ. noleap) unit_code = -13
      ENDIF

* Named calendar with unit=month or year gets length according to calendar's year length.

        true_month = .FALSE. ! initialize
	unit_code = TM_UNITS_CAL( unit_code, agg_cal_id, true_month )

        IF ( unit_code.EQ.0 .AND. tunits .NE. ' ' ) GOTO 5100

* successful completion
        status = merr_ok
        RETURN

* error exits
 5000   RETURN
 5100   CALL TM_ERRMSG (merr_badlinedef, status, 'CD_GET_T0',
     .                  no_descfile, filename(:flen),
     .                  'unintelligible units '//units_string,
     .                  no_errstring, *5000)
        END
