	SUBROUTINE TSTEP_TO_DATE ( grid, tstep, precision, date )

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

* Arguments:
*	grid	 - grid containing a time axis
*	tstep    - time step
*	precision- precision (1 to 6 for day,mon,yr,hr,min,sec, respectively )
*			(use ABS(precision) - neg. indicates truncate left 0's)
*	date	 - returned string containing date in ASCII form
*
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations
	CHARACTER*(*)	date
	INTEGER		grid, precision
	REAL*8		tstep
*
* Local definitions
	CHARACTER*20	SECS_TO_DATE, buff
	INTEGER		axis, status, prec
	REAL*8		start_secs, offset_secs, this_secs
	REAL*8		SECS_FROM_BC

!	REAL*8		start_secs, offset_secs, this_secs, prec_fact(6)
!	DATA	prec_fact	/  31536000.,  2628000.,  86400.,  3600.,
!     .				   60., 1./

* initialize
	prec = ABS(precision)

* get the axis number
	axis = grid_line( t_dim, grid )
	IF ( axis .EQ. mnormal .OR. axis .EQ. munknown ) THEN
	   WRITE ( date, * ) SNGL(tstep)
	   RETURN
	ELSEIF ( line_direction(axis) .NE. 'TI' ) THEN
	   WRITE ( date, * ) SNGL(tstep)
	   RETURN
	ENDIF

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	start_secs = SECS_FROM_BC( line_t0(axis), status )

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*line_tunit( axis )

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* round off date instead of truncating 
* ( people don't normally view dates this way )
!	this_secs = ( DNINT(this_secs/prec_fact(prec)) + 0.5 ) * prec_fact(prec)

* Convert to date
	buff = SECS_TO_DATE(this_secs)

* Select the requested precision
	IF ( prec .GT. 6 ) THEN		! full precision
	   date = buff
	ELSEIF ( prec .GE. 3 ) THEN	! dd-mmm-$_...
	   date = buff( :date_str_len(prec) )
	ELSEIF ( prec .EQ. 2 ) THEN	! mmm-$_
	   date = buff( 4:11 )	
	ELSE				! $_
	   date = buff( 8:11 )	
	ENDIF

* Finished
	RETURN
	END
