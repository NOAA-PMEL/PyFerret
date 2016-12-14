	SUBROUTINE TM_MONTH_TIME_FACTOR (grid, true_month, timefac)


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
* Return a scale factor to write true-monthly data in original time units. 
* Converting from the secs-per-month in the true-month calendar to units from 
* the original axis.
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

	include 'ferret.parm'
	include 'tmap_dims.parm'
        include 'xunits.cmn_text'
	include 'calendar.decl'
        include 'xtm_grid.cmn_text'
        external xgt_grid_data

	LOGICAL true_month
	INTEGER grid
	REAL timefac

	LOGICAL ITSA_TRUEMONTH_AXIS
	INTEGER TM_GET_CALENDAR_ID, 
     .		iaxis, cal_id, monax_units_code, status
	REAL	secspmon, secs_p_unit, unit_p_mon

	timefac = 1.
	iaxis = grid_line(t_dim, grid)
!	print *, line_units(iaxis)

* get axis description stored by Ferret

	true_month = ITSA_TRUEMONTH_AXIS(iaxis)
	IF (.NOT. true_month) RETURN

	monax_units_code = line_unit_code(iaxis) 

	cal_id = TM_GET_CALENDAR_ID ( line_cal_name(iaxis) )

* Convert to days

* Translation from line-unit-code (true month) to days

	secspmon = un_convert( monax_units_code )

	secs_p_unit = un_convert( pun_day )
	unit_p_mon = secspmon/ secs_p_unit
	timefac = unit_p_mon

	RETURN
	END
