	SUBROUTINE EQUAL_RANGE_T( string,idim,grid,lo,hi,delta,status )

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
* if the string is simply "name" then return unspecified values
* the "delta" value is optional - return unspecfied_val8 if not given

* This routine differs from EQUAL_RANGE in the treatment of time axes.
* This routine always returns a tstep value.  If MODE CALENDAR is TRUE
* this routine checks to make sure the points are ordered.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V3.00  20/10/93 - *sh*

* calling argument declarations
	INTEGER		idim, grid, status
	REAL*8		lo, hi, delta
	CHARACTER*(*)	string

* internal variable declarations
	LOGICAL		GEOG_LABEL, formatted
	REAL*8		SECS_TO_TSTEP, tmp

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
#endif

* first get EQUALL_RANGE to convert the data normally
	CALL EQUAL_RANGE( string, idim, lo, hi, delta,
     .			  formatted, status )
	IF ( status .NE. ferr_OK ) RETURN

	IF ( idim .EQ. t_dim ) THEN
* if the time was given as a date convert it to a tstep
	   IF ( formatted ) THEN
	      lo = SECS_TO_TSTEP( grid, lo )
	      hi = SECS_TO_TSTEP( grid, hi )
	   ENDIF		

* if MODE CALENDAR is TRUE in FERRET make sure the times are ordered
	   IF ( GEOG_LABEL(t_dim,grid)
     .    .AND. lo .GT. hi		 ) THEN 
	      tmp = lo
	      lo = hi
	      hi = tmp
	   ENDIF
	ENDIF

* success
 1000	RETURN
	END
