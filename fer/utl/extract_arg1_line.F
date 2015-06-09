	SUBROUTINE EXTRACT_ARG1_LINE( src, axis, axmin, axmax,
     .				      i, j, k, l, m, n, line )

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
* extract a single line of data along requested axis from a memory variable
* Note that whatever the axis orientation, the corresponging index position
* from the i,j,k,l,m,n arguments is irrelevant

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V698+ *sh* 5/15

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	axis, axmin, axmax, i, j, k, l, m, n
        REAL    src(m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .              m1lot:m1hit,m1loe:m1hie,m1lof:m1hif )
	REAL    line(axmin:axmax)

* internal variable declarations:
	INTEGER	ss

* extract the line
	IF (axis .EQ. x_dim) THEN
	   DO ss = axmin, axmax
	      line(ss) = src(ss, j, k, l, m, n)
	   ENDDO
	ELSEIF (axis .EQ. y_dim) THEN
	   DO ss = axmin, axmax
	      line(ss) = src(i, ss, k, l, m, n)
	   ENDDO
	ELSEIF (axis .EQ. z_dim) THEN
	   DO ss = axmin, axmax
	      line(ss) = src(i, j, ss, l, m, n)
	   ENDDO
	ELSEIF (axis .EQ. t_dim) THEN
	   DO ss = axmin, axmax
	      line(ss) = src(i, j, k, ss, m, n)
	   ENDDO
	ELSEIF (axis .EQ. e_dim) THEN
	   DO ss = axmin, axmax
	      line(ss) = src(i, j, k, l, ss, n)
	   ENDDO
	ELSE    ! f_dim
	   DO ss = axmin, axmax
	      line(ss) = src(i, j, k, l, m, ss)
	   ENDDO
	ENDIF

	RETURN
	END
