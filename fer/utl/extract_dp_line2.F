	SUBROUTINE EXTRACT_DP_LINE2( cx, grid_data, dp_line, idim, nout )

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
* extract a single line of data from memory variable #2, grid along axis idim
* no error checking is done

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* from extract_dp_line.F
*
* V570 - reading 2nd line for bounds on a DEFINE AXIS/BOUNDS
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)

* 5/15 NOTE: the newer routine EXTRACT_ARG2_LINE can probably replace this
*      and is more appropriately sensitive to argument idim
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 


        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
	include 'xmem_subsc.cmn'

* calling argument declarations:
	INTEGER	cx, mv, idim, nout
	REAL    grid_data( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit,m1loe:m1hie,m1lof:m1hif )
	REAL*8	dp_line(nout)

* internal variable declarations:
	INTEGER	iout, i, j, k, l, m, n

* extract the line
	iout = 0
	DO n = cx_lo_s6(cx), cx_hi_s6(cx)
	DO m = cx_lo_s5(cx), cx_hi_s5(cx)
	DO l = cx_lo_s4(cx), cx_hi_s4(cx)
	DO k = cx_lo_s3(cx), cx_hi_s3(cx)
	DO j = cx_lo_s2(cx), cx_hi_s2(cx)
	DO i = cx_lo_s1(cx), cx_hi_s1(cx)

	   iout = iout + 1
	   dp_line( iout ) = grid_data( i,j,k,l,m,n )

	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
