	SUBROUTINE EF_GET_DATE_TSTEP ( grid, dim, tstep, precision, date )
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
* This is an older version of TSTEP_TO_DATE which was revised for better
* output formatting in Ferret v6.0.  This version gives a fixed unchanging
* format.
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* v75  *acm* 3/18  Changes to allow for timesteps in fractional seconds
* v74+ *acm* 12/18 Issue 1902, fractional seconds only when time units are seconds
*                  use AINT call not INT when comparing values, large numbers of seconds
* v745 *acm* 12/18  Issue 1909, option to write dates in ISO8601 format
* v7441*acm* 12/18 Issue 1910: Allow dates in commands to be in ISO 8601 form, yyyy-mm-dd

* Arguments:
*	grid	 - grid containing a time axis
*	dim      - dimension (t_dim or f_dim)
*	tstep    - time step
*	precision- precision (1 to 6 for day,mon,yr,hr,min,sec, respectively )
*			(use ABS(precision) - neg. indicates truncate left 0s)
*	date	 - returned string containing date in ASCII form
*

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
        include 'xunits.cmn_text'
	include 'xtext_info.cmn'

* calling argument declarations
	CHARACTER*(*)	date
	INTEGER		grid, dim, precision
	REAL*8		tstep
*
* Local definitions
        LOGICAL         ITSA_TRUEMONTH_AXIS, modulo
	CHARACTER*30	SECS_TO_DATE_OUT, buff
	CHARACTER*48	TM_FMT, buff2
	CHARACTER*2     linedir
	INTEGER		TM_GET_CALENDAR_ID, TM_LENSTR1,  
     .                  axis, status, prec, cal_id, nfrac, slen
	REAL*8		start_secs, offset_secs, this_secs, val
	REAL*8		SECS_FROM_BC

!	REAL*8		start_secs, offset_secs, this_secs, prec_fact(6)
!	DATA	prec_fact	/  31536000.,  2628000.,  86400.,  3600.,
!     .				   60., 1./

* initialize
	prec = ABS(precision)
        linedir = 'TI'
	IF (dim .EQ. f_dim) linedir = 'FI'  ! formatted forecast date

* get the axis number
	axis = grid_line( dim, grid )
	modulo = line_modulo(axis)

	IF ( axis .EQ. mnormal .OR. axis .EQ. munknown ) THEN
	   WRITE ( date, * ) (tstep)
	   RETURN
	ELSEIF ( line_direction(axis) .NE. linedir ) THEN
	   WRITE ( date, * ) (tstep)
	   RETURN
	ENDIF

* get the calendar type

        cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	start_secs = SECS_FROM_BC( line_t0(axis), cal_id, status )

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*line_tunit( axis )
	IF (ITSA_TRUEMONTH_AXIS(axis)) offset_secs = tstep* un_convert(pun_day)  

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* round off date instead of truncating 
* ( people do not normally view dates this way )
!	this_secs = ( DNINT(this_secs/prec_fact(prec)) + 0.5 ) * prec_fact(prec)

* Convert to date

	buff = SECS_TO_DATE_OUT(this_secs, cal_id, modulo, prec)
*       SECS_TO_DATE_OUT has already dealt with only printing 
*       the appropriate values for precision 1 to 6
	date = buff

* If fractional seconds are requested....
	IF ( prec .GT. 6 ) THEN	
	   IF (line_tunit(axis) .EQ. 1) THEN  ! Deal with fractional seconds
	      val = AINT(tstep)
	      IF (val .NE. tstep) THEN
	         val = tstep - val
	         buff2 = TM_FMT(val, 4, 16, nfrac)
		 date = buff(:TM_LENSTR1(buff))//buff2(2:nfrac)
	      ELSE 
	         date = buff(:TM_LENSTR1(buff))//'.0'
	      ENDIF
	   ENDIF
	ENDIF

	slen = TM_LENSTR1(date)
	CALL CHECK_DATE_REFORMAT (date, slen, cal_id, status)

* Finished
	RETURN
	END

