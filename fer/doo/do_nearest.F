	SUBROUTINE DO_NEAREST(	idim, by_index, find_above,
     .					com, com_mr, com_cx,
     .					res, res_mr, res_cx )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* for each point, locate the valid point along the specified axis
* that is nearest to the point

* This function performs 4 transformations
*       @CDA - closest distance above (world coord distance)
*       @CDB - closest distance below (world coord distance)
*       @CDA - closest index    above
*       @CDA - closest index    below

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V450: 8/97
* v491 - *kob* 3.98 - modify loop @450 to use cx_lo/hi_s3 for k 
*                     rather than s4 *kob*
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xcontext.cmn'
	include	'xvariables.cmn'
	include	'xmem_subsc.cmn'

* calling argument declarations:
	LOGICAL by_index, find_above
	INTEGER	idim, action, com_mr, com_cx, res_mr, res_cx
        REAL    com( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          res( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* internal variable declarations:
	LOGICAL TM_ITSA_DSG
	INTEGER i, j, k, l, m, n, grid, uplim, dnlim, offlim,
     .		delta, las_index
	REAL	bad_com, bad_res
	REAL*8  TM_WORLD
	CHARACTER*150 VAR_TRANS, buff

* --- end of introductory code ---

* initialize
	grid = mr_grid( com_mr )
	bad_com = mr_bad_data( com_mr )
	bad_res = mr_bad_data( res_mr )


* These not customized for DSG data, not sure they really make sense to apply. Write a note.

	IF ( TM_ITSA_DSG(grid) ) THEN 
	   buff = VAR_TRANS(idim, res_cx, j)
	   i = INDEX(buff, '@')

	   CALL WARN ('Transformation '//buff(i:i+4)//
     .     ' not customized for Discrete Sampling Geometries: computed over all features.')
	ENDIF


* determine axis limits along axis of transformation
	IF ( find_above ) THEN		     ! @NRDA and @NRIA
	  offlim = cx_hi_ss(com_cx, idim)    ! source data beyond edge
	  uplim = cx_lo_ss(res_cx, idim)
	  dnlim = cx_hi_ss(res_cx, idim)
	  delta = -1
	ELSE				     ! @NRDB and @NRIB
	  offlim = cx_lo_ss(com_cx, idim)    ! source data beyond edge
	  uplim = cx_hi_ss(res_cx, idim)
	  dnlim = cx_lo_ss(res_cx, idim)
	  delta =  1
	ENDIF

* ALONG X AXIS
	IF ( idim .EQ. x_dim ) THEN
	   DO n = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	   DO m = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	   DO l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO i = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = i
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO i = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = i
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(i-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(i,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Y AXIS
	ELSEIF ( idim .EQ. y_dim ) THEN
	   DO n = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	   DO m = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	   DO l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO j = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = j
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO j = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = j
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(j-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(j,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG Z AXIS
	ELSEIF ( idim .EQ. z_dim ) THEN
	   DO n = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	   DO m = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	   DO l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO k = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = k
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO k = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = k
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(k-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(k,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG T AXIS
	ELSEIF ( idim .EQ. t_dim ) THEN
	   DO n = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	   DO m = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	   DO k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO l = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = l
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO l = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = l
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(l-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(l,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

* ALONG E AXIS
	ELSEIF ( idim .EQ. e_dim ) THEN
	   DO n = cx_lo_s6(res_cx), cx_hi_s6(res_cx)
	   DO l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO m = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = m
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO m = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = m
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(m-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(m,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
 
* ALONG F AXIS
	ELSEIF ( idim .EQ. f_dim ) THEN
	   DO m = cx_lo_s5(res_cx), cx_hi_s5(res_cx)
	   DO l = cx_lo_s4(res_cx), cx_hi_s4(res_cx)
	   DO k = cx_lo_s3(res_cx), cx_hi_s3(res_cx)
	   DO j = cx_lo_s2(res_cx), cx_hi_s2(res_cx)
	   DO i = cx_lo_s1(res_cx), cx_hi_s1(res_cx)
* ... locate first valid index off edge of requested region
	      las_index = unspecified_int4
	      DO n = offlim, dnlim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) las_index = n
	      ENDDO
* ... loop through points finding distance from last valid point
	      DO n = dnlim, uplim, delta
	         IF ( com(i,j,k,l,m,n) .NE. bad_com ) THEN
	            las_index = n
	            res(i,j,k,l,m,n) = 0.0
	         ELSEIF ( las_index.EQ. unspecified_int4 ) THEN
	            res(i,j,k,l,m,n) = bad_res
	         ELSEIF ( by_index ) THEN
* ... distance in index space
	            res(i,j,k,l,m,n) = ABS(n-las_index)
	         ELSE
* ... distance in world coordinates
	            res(i,j,k,l,m,n) = ABS(
     .                    TM_WORLD(n,         grid, idim, box_middle)
     .                  - TM_WORLD(las_index, grid, idim, box_middle) )
	         ENDIF
	      ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO
	   ENDDO

	ENDIF

	RETURN
	END

