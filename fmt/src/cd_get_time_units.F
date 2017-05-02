	SUBROUTINE CD_GET_TIME_UNITS (buff, cal_id, units_str, 
     .			              t0_str, days_neg4713, status)


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
* 2/2017 acm
* Get the time units from a string 'since date', set units_str
* and t0_str if the string is valid.
* 
* The return days_neg4713 is returned non-zero if the axis is from 
* a Matlab-generatedfile with "Julian astronomical date".
*
* Code taken fron cd_get_1_axis.F, new routine for ticket 2246;
* Drawing formatted time axes on a PLOT/VS plot.

        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'xunits.cmn_text'
        external xunits_data
#include "gt_lib.parm"
        include 'calendar.decl'
        include 'calendar.cmn'

	CHARACTER*(*) buff, units_str, t0_str
	INTEGER cal_id, status
	REAL*8	days_neg4713

	CHARACTER*3 dbuf
	INTEGER	TM_LENSTR1, STR_UPCASE, since_T0, slen, istat
	REAL*8	TM_SECS_FROM_BC, secs_to_1901, secs_to_1968, 
     .		days_1901_may68 
	
* parameter declarations
        REAL*8          psecs_per_day
        PARAMETER     ( psecs_per_day = 60.*60.*24. )

	since_T0 = MAX( INDEX(buff,'since'), INDEX(buff,'SINCE') )
	units_str = buff(:since_T0-1)  ! just the "tunits" part 

	istat = STR_UPCASE( dbuf, buff(1:3) )

	slen = TM_LENSTR1(buff)

	IF (buff(since_T0+6:since_T0+10) .EQ. '-4713' .AND.
     .      dbuf .EQ. 'DAY')  THEN

c  Matlab Julian date encoding
c  This is the simple conversion; but we will reset the origin to 1901.
c       buff = 'DAYS since 1968-05-23 00:00:00'
c       days_neg4713 = 2440000.D0

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
     .                            cal_id, t0_str, 
     .                            pdate_vax, .FALSE., status )

	RETURN
	END
