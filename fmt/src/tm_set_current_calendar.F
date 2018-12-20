      SUBROUTINE TM_SET_CURRENT_CALENDAR (set_name, status)

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
C     Set the current calendar name and id in common current_calendar
C     and loads calendar info into common.
C
C     Returns merr_varlim if calendar is invalid, merr_ok successful completion
C     J Davison 10.99
C     V 1.0

C  11/29/00  Ansley Manke  revise to match just 3 characters of allowed_calendars
* V6.74 3/12 *acm* cleanup: add disclaimer
* V7.45 12/18 *acm* remove dependence on string length of allowed_calendars

C     0.00000000000000000000000000000000000000000000000000000000000000001

      include 'calendar.decl'
      include 'calendar.cmn'
      include 'tmap_errors.parm'
 
      integer         end, ndx, status
      character*30    test_string

      character*(*)   set_name

*************************************************************************

C     Move input string to upper case
      CALL upnsquish (set_name,test_string,end)

C     LOCATE NAME IN CALENDAR TABLE
      DO 10 ndx = 1, max_calendars
         IF (test_string(1:3) .EQ. allowed_calendars(ndx)(1:3)) 
     .        GOTO 100
 10   END DO

C     error: calendar not found

      status = merr_varlim
      goto 1000

C     calendar found, set current calendar

 100  CONTINUE

      CALL tm_load_calendar (allowed_calendars(ndx), status)

 1000 RETURN

      END

