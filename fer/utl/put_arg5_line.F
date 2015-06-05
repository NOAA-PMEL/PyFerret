	SUBROUTINE PUT_ARG5_LINE( src, axis, axmin, axmax, offset,
     .				  i, j, k, l, m, n, line )

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
* put a single line of data along designateded axis into a memory variable
* Note that whatever the axis orientation, the corresponging index position
* from the i,j,k,l,m,n arguments is irrelevant
* The offset arg is a hack to allow the index limits on thesrc and line to
* be offset from one another by a constant 

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V698+ *sh* 5/15 -- derived from EXTRACT_ARG1_LINE

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	axis, axmin, axmax, offset, i, j, k, l, m, n
        REAL    src(m5lox:m5hix,m5loy:m5hiy,m5loz:m5hiz,
     .              m5lot:m5hit,m5loe:m5hie,m5lof:m5hif )
	REAL    line(axmin:axmax)

* internal variable declarations:
	INTEGER	ss

* extract the line
	IF (axis .EQ. x_dim) THEN
	   DO ss = axmin, axmax
	      src(ss-offset, j, k, l, m, n) = line(ss)
	   ENDDO
	ELSEIF (axis .EQ. y_dim) THEN
	   DO ss = axmin, axmax
	      src(i, ss-offset, k, l, m, n) = line(ss)
	   ENDDO
	ELSEIF (axis .EQ. z_dim) THEN
	   DO ss = axmin, axmax
	      src(i, j, ss-offset, l, m, n) = line(ss)
	   ENDDO
	ELSEIF (axis .EQ. t_dim) THEN
	   DO ss = axmin, axmax
	      src(i, j, k, ss-offset, m, n) = line(ss)
	   ENDDO
	ELSEIF (axis .EQ. e_dim) THEN
	   DO ss = axmin, axmax
	      src(i, j, k, l, ss-offset, n) = line(ss)
	   ENDDO
	ELSE    ! f_dim
	   DO ss = axmin, axmax
	      src(i, j, k, l, m, ss-offset) = line(ss)
	   ENDDO
	ENDIF

	RETURN
	END
