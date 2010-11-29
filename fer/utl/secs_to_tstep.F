	DOUBLE PRECISION FUNCTION SECS_TO_TSTEP( grid, secs )

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
* convert an absolute time measured in seconds from 1-JAN-0000:00:00:00
* into a time step value relevant on a particular grid, "grid"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/27/88
* V530: *acm* 12/00 add calendar types; cal_id

#ifdef unix
	include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'
	external xgt_grid_data
        include 'ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	grid
	REAL*8	secs

* internal variable declarations:
	INTEGER		TM_GET_CALENDAR_ID, axis, cal_id, status
	CHARACTER*20	date
	REAL*8		SECS_FROM_BC, bc_to_t0

* determine time axis of grid
	axis = grid_line( t_dim, grid )

* valid time axis ?
	IF ( line_direction( axis ) .EQ. 'TI' ) THEN

* get calendar type

	   cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )

* seconds from BC to start of time axis
	   bc_to_t0 = SECS_FROM_BC( line_t0(axis), cal_id, status )

* convert to units used on this time axis
	   SECS_TO_TSTEP = ( secs - bc_to_t0 ) / line_tunit(axis)

	ELSE

* not a valid time axis
	   SECS_TO_TSTEP = secs		! no conversion possible

	ENDIF
	RETURN
	END
