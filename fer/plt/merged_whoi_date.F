	CHARACTER*14 FUNCTION MERGED_WHOI_DATE( cx_list, ncx )

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
* create a string of the form YYMMDDHHMMSSCC (internal ordering) based
* on the earlies time to be plotted from the given contexts

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/27/95 - 4-digit YEARS!!
* V530:  1/01 *acm* cal_id for alternative calendar axis types

*#ifdef unix
*#	include 'ferret.parm'
*#	include 'xcontext.cmn'

#ifdef unix
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	ncx, cx_list(ncx)

* internal variable declarations
	INTEGER 	yr100, yr, mon, day, hr, mn, sec,
     .			status, i, grid, cx, axis
        INTEGER		TM_GET_CALENDAR_ID, cal_id
	CHARACTER*20	TM_SECS_TO_DATE, date
	REAL*8		TSTEP_TO_SECS, bcsecs

* internal parameter declarations
	REAL*8		yrofsecs
	PARAMETER     ( yrofsecs = 60.*60.*24.*365. )

* determne the earliest date to be plotted
	DO 100 i = 1, ncx
	   cx = cx_list(i)
	   grid = cx_grid(cx)
	   IF ( i .EQ. 1 ) THEN
	      bcsecs = TSTEP_TO_SECS( grid, cx_lo_ww(t_dim,cx) )
	   ELSE
	      bcsecs = MIN( bcsecs,
     .		       TSTEP_TO_SECS( grid, cx_lo_ww(t_dim,cx) ) )
	   ENDIF

* Convert to date
          axis = grid_line ( t_dim, grid)
          cal_id = TM_GET_CALENDAR_ID(line_cal_name(axis) )

* arbitrarily subtract a year -- needed cuz PLOT* rounds down the "tmin" value
 	  bcsecs = MAX( 0.0D0, bcsecs - yrofsecs )
	  date = TM_SECS_TO_DATE(bcsecs, cal_id)

 100	CONTINUE

* break the date string into component integer values
	CALL TM_BREAK_DATE( date, cal_id, 
     .                      yr, mon, day, hr, mn, sec, status )
	yr100 = yr/100		! 1980 --> 19
	yr = yr - (yr100*100)	! 1982 --> 82

* assemble the pieces in WHOI format
	WRITE (MERGED_WHOI_DATE,3000) yr,mon,day,hr,mn,sec,yr100
 3000	FORMAT (7I2.2)

	RETURN
	END
