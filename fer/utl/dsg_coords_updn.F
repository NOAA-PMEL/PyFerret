	SUBROUTINE DSG_COORDS_UPDN (npts, coords, bad, ascending, status)
	      

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
* Are the coordinates in ascending or decending order?

* NOAA/PMEL, Seattle, WA - Science Data Integration Group
* V751 8/19 *acm* 
*     11/2022 *acm* Code cleanup: renamed the file to match the routine name 
*                   (was dsg_coord_updn.F)

        IMPLICIT NONE
	include 'ferret.parm'
	include 'errmsg.parm'

* calling argument declarations:
	LOGICAL ascending
	INTEGER npts, status
	REAL	coords(*), bad

* internal variable declarations:
        INTEGER i, ic, change
	REAL	val0

	status = ferr_ok

* ... find first valid value

	ic = 1
	DO i = 1, npts
	   val0 = coords(1)
	   ic = i
	   IF (val0 .NE. bad) EXIT
	ENDDO

* ... all bad, or only the last one valid (shouldn't happen)
	IF (ic .EQ. npts) RETURN

* ... Are the values ascending or descending? 
*     If they change more than once, that's an error

	change = 0
	DO i = ic+1, npts
	   IF (coords(i).LT.val0 .AND. ascending) THEN
	      ascending = .FALSE.
	      change = change+1
	      IF (change .GT. 1) GOTO 5100
	   ENDIF

	   IF (coords(i).GT.val0 .AND. .NOT.ascending) THEN
	      ascending = .TRUE.
	      change = change+1
	      IF (change .GT. 1) GOTO 5100
	   ENDIF
	   val0 = coords(i)
	ENDDO

 5000	RETURN

 5100	CALL ERRMSG( ferr_regrid, status,
     .		'Z coordinates within profiles are not monotonic', *5000)
	END
