	SUBROUTINE CLEAN_COORD_STRING ( idim, string )

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
* axe the time field from climatological dates: "dd-mmm      hh:mm" -> "dd-mmm"
* replace 19E(379) with the unambiguous 379E

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V300:  8/27/93

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER idim
	CHARACTER*(*) string

* internal variable declarations:
	INTEGER		TM_LENSTR1, i, i0, slen

	IF ( idim.EQ.y_dim .OR. idim.EQ.z_dim ) RETURN

	slen = TM_LENSTR1( string )

	IF ( idim .EQ. t_dim ) THEN
* FIX TIMES
	   IF ( slen .LE. 11 ) RETURN   	! date, only - no time
	   IF ( string(3:3) .NE. '-' ) RETURN	! not a date string

	   IF ( string(9:9) .EQ. ' ' ) THEN
* wipe out time on climatological dates
	      DO 100 i = 7, slen
	        string(i:i) = ' '
 100	      CONTINUE
	   ELSE
	     string(12:12) = ':'
	   ENDIF

	ELSE
* FIX LONGITUDES
	   i0 = INDEX( string, '(' )      ! e.g."19E(379)"
	   IF ( i0 .GT. 0 ) THEN
	      DO 200 i = i0+1, slen-1     ! --> "379(379)"
 200	      string(i-i0:i-i0) = string(i:i)
	      DO 210 i = slen-i0, slen    ! --> "379     "
 210	      string(i:i) = ' '
	      slen = slen-i0
	      string(slen:slen) = 'E'     ! --> "379E"
	   ENDIF

	ENDIF

	RETURN
	END
