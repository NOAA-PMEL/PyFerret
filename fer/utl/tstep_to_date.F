	SUBROUTINE TSTEP_TO_DATE ( grid, idim, tstep, precision, date )

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
* Return a string giving the date for a particular timestep
* This is done by figuring the offset from
* 01-JAN-0000 00:00:00 in seconds, and then calculating the date.
*
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* rev. 0.0 3/17/88 - based on TM_STEP_TO_DATE by *mv*
* rev. 0.1 - 2/16/89 - TM_SECS_TO_DATE --> SECS_TO_DATE
* V300 4/93 - crash if precision exceeds 6
* V530: *acm* 12/00 add calendar types; cal_id
* V600 - SECS_TO_DATE --> SECS_TO_DATE_OUT for better formatting
* V610: *acm*  2/08 Fix bug 762: if the axis is not modulo, always list the
*                   year (previously never listed year 0 or year 1)
*       *acm* 3/12 6D Ferret: Add E and F dimensions 
*                  A time axis may be a T axis or F axis.
* V698  *acm* 5/16 Ticket 2405: MODE CALENDAR:auto
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.

* Arguments:
*	grid	 - grid containing a time axis
*	tstep    - time step
*	precision- precision (1 to 6 for day,mon,yr,hr,min,sec, respectively )
*			(use ABS(precision) - neg. indicates truncate left 0's)
*	date	 - returned string containing date in ASCII form
*

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xtext_info.cmn'
        include 'xunits.cmn_text'
        external xunits_data

* calling argument declarations
	CHARACTER*(*)	date
	INTEGER		grid, idim, precision
	REAL*8		tstep
*
* Local definitions
        LOGICAL         ITSA_TRUEMONTH_AXIS, modulo
	CHARACTER*20	SECS_TO_DATE_OUT, buff, dms
	INTEGER		TM_GET_CALENDAR_ID, DATE_PREC, axis, status, prec, 
     .                  cal_id
	REAL*8		start_secs, offset_secs, this_secs, delta
	REAL*8		SECS_FROM_BC

!	REAL*8		start_secs, offset_secs, this_secs, prec_fact(6)
!	DATA	prec_fact	/  31536000.,  2628000.,  86400.,  3600.,
!     .				   60., 1./

* initialize
	prec = ABS(precision)

* get the axis number

	axis = grid_line( idim, grid )
	IF ( axis .EQ. mnormal .OR. axis .EQ. munknown ) THEN
	   WRITE ( date, * ) (tstep)
	   RETURN
	ELSEIF ( (line_direction(axis).NE.'TI') .AND.
     .           (line_direction(axis).NE.'FI')  ) THEN
	   WRITE ( date, * ) (tstep)
	   RETURN
	ENDIF

* get the calendar type

        cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	start_secs = SECS_FROM_BC( line_t0(axis), cal_id, status )

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*line_tunit( axis )

* True-month axis? A true-month step has been converted from month to days
	IF (ITSA_TRUEMONTH_AXIS(axis)) offset_secs = tstep* un_convert(pun_day)  

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* round off date instead of truncating 
* ( people don't normally view dates this way )
!	this_secs = ( DNINT(this_secs/prec_fact(prec)) + 0.5 ) * prec_fact(prec)

*** Do all this higher up for plots, as is done for listings, where date_prec is
*** called in row_coord_fmt.

* Will write only month-day in labels for monthly climatology
        modulo = line_modulo(axis)

* Convert to date
	date = SECS_TO_DATE_OUT(this_secs, cal_id, modulo, prec)


* Finished
	RETURN
	END
