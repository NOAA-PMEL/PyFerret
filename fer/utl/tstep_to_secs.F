	DOUBLE PRECISION FUNCTION TSTEP_TO_SECS( grid, tstep )

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
* convert a time step value relevant on a particular grid, "grid" into an
* absolute time measured in seconds from 1-JAN-0000:00:00:00

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/27/88
* V320: 12/94 - bug fix: not based on TSTEP_TODATE because the returned
*	"date" string blanks out years 0000 and 0001 ==> cannot distinguish

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	grid
	REAL*8	tstep

* internal variable declarations:
	INTEGER		axis, status
!	CHARACTER*20	date
	REAL*8		SECS_FROM_BC, start_secs, offset_secs

* internal parameter declarations:
	INTEGER		seconds
	PARAMETER     ( seconds = 6 )

* determine time axis of grid
	axis = grid_line( t_dim, grid )

* valid time axis ?
	IF ( line_direction( axis ) .EQ. 'TI' ) THEN

! pre 12/94 algorithm
!* convert time step to date of format dd-mmm-yyyy:hh:mm:ss
!	   CALL TSTEP_TO_DATE ( grid, tstep, seconds, date )
!
!* convert date to seconds since 1-JAN-0000:00:00:00
!	   TSTEP_TO_SECS = SECS_FROM_BC( date, status )

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file
	   start_secs = SECS_FROM_BC( line_t0(axis), status )

* Find the number of seconds since start of data set for this time step
	   offset_secs = tstep*line_tunit( axis )

* Total seconds to this time step since 01-JAN-0000 00:00:00
	   TSTEP_TO_SECS = start_secs + offset_secs

	ELSE

* not a valid time axis
	   TSTEP_TO_SECS = tstep

	ENDIF
	RETURN
	END
