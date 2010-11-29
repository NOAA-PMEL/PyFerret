	CHARACTER*14 FUNCTION INTERNAL_WHOI_DATE( grid, tstep )

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
* create a string of the form YYMMDDHHMMSSCC (internal ordering) from the
* time step

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/15/88
* revision 0.1 - 1/18/89 - BREAK_DATE --> TM_BREAK_DATE
* V200:  1/31/90 - fixed bug with climatological axes (years "0000" and "0001"
*		   but produce the same WHOI dates)
* V420: 11/27/95 - 4-digit YEARS!!
* v530  12/19/00 *acm* cal_id for non-standard calendar types.

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

* calling argument declarations
	INTEGER	grid
	REAL*8	tstep

* internal variable declarations
	INTEGER 	yr100, yr, mon, day, hr, min, sec,
     .			status, i, axis
	INTEGER		TM_GET_CALENDAR_ID, cal_id
	CHARACTER*20	TM_SECS_TO_DATE, date
	REAL*8		start_secs, offset_secs, this_secs
	REAL*8		SECS_FROM_BC

* get the axis number
	axis = grid_line( t_dim, grid )
	IF ( axis .EQ. mnormal .OR. axis .EQ. munknown ) THEN
	   INTERNAL_WHOI_DATE = '00000000000000'
	   RETURN
	ELSEIF ( line_direction(axis) .NE. 'TI' ) THEN
	   INTERNAL_WHOI_DATE = '00000000000000'
	   RETURN
	ENDIF

* First find the offset from 01-JAN-0000 00:00:00 from the descriptor file

        cal_id = TM_GET_CALENDAR_ID ( line_cal_name(axis) )
	start_secs = SECS_FROM_BC( line_t0(axis), cal_id, status )

* Find the number of seconds since start of data set for this time step
	offset_secs = tstep*line_tunit( axis )

* Total seconds to this time step since 01-JAN-0000 00:00:00
	this_secs = start_secs + offset_secs

* Convert to date
	date = TM_SECS_TO_DATE(this_secs, cal_id)

* break the date string into component integer values
	CALL TM_BREAK_DATE( date, cal_id, yr, mon, day, hr, min, 
     .                      sec, status )
	yr100 = yr/100		! 1980 --> 19
	yr = yr - (yr100*100)	! 1982 --> 82

* assemble the pieces in WHOI format
	WRITE (INTERNAL_WHOI_DATE,3000) yr,mon,day,hr,min,sec,yr100
 3000	FORMAT (7I2.2)

	RETURN
	END
