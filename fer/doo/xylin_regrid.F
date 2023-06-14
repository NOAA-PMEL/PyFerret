	SUBROUTINE XYLIN_REGRID( cx_lims,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, src_cx, dst_cx, 
     .			         isrc_lo, isrc_hi,
     .			         iss12, coefi, jss12, coefj, work)

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
* 	Using data values from src organized on grid src_grid, interpolate to
* 	values dst on grid dst_grd.

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which

* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

* 	Both src and dst are assumed to be 2 dimensional structures
* 	lying in the same plane - given by mr_DIM1 and mr_DIM2.

*	ss12i and sst12j hold the indices of nearest grid nodes on src_grd 
*       for nodes on dst_grd. coefi and coefj hold the interpolation coefficients.

* First just replicate what Ferret has always done: first linear regrid in y then
* in x.  Also program a bilinear regridding - this will let us do more to fill in
* gaps in the future, but we will need to figure out how to decide what the handling
* of missing data should be.

* programmer - Ansley Manke, from xyave_regrid.F
* NOAA/PMEL, Seattle, WA - TMAP
*
* 6/22/2011
*                    The XY linear regridding matches past Ferret behavior, 
*                    not using bilinear regridding for now
* V68  *acm* 1/12   use TM_FPEQ for double-precision ferret.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm);
*                   removed unused bilinear interpolation code.
* V702 *sh* 3/2017 Dynamic memory revealed a bug in array "work" dimensioning
*		   It was dimensioned using the mr size instead of the cx
*		   which is wrong when a larger object is re-used from cache
*                  To address it we turned isrc_lo, isrc_hi into call arguments
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, msrc, mdst, dst_lox, dst_hix,
     .          dst_loy, dst_hiy, src_cx, dst_cx,
     .	        isrc_lo, isrc_hi 
	INTEGER	iss12(dst_lox:dst_hix+1), jss12(dst_loy:dst_hiy+1)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .		coefi( dst_lox:dst_hix ), coefj(dst_loy:dst_hiy)
        REAL    work(isrc_lo:isrc_hi,dst_loy:dst_hiy)  ! srcx by dsty

* local variable declarations:

        LOGICAL	TM_FPEQ
	INTEGER	dim1, dim2, src_grid, dst_grid,
     .          i, j, k, l, m, n, iss, jss,
     .                            jsrc_lo, jsrc_hi,
     .          idst_lo ,idst_hi, jdst_lo, jdst_hi
	REAL	bad_src, bad_dst, c1i, c1j, c2i, c2j, 
     .          z1, z2
********************************************************************************

* initialize

* source and destination grids
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )
	dim1     = x_dim
	dim2     = y_dim

* limits for calculation
!        isrc_lo = cx_lo_ss(src_cx,dim1)	! 3/17 passed as arg
!        isrc_hi = cx_hi_ss(src_cx,dim1)	! 3/17 passed as arg
        jsrc_lo = cx_lo_ss(src_cx,dim2)
        jsrc_hi = cx_hi_ss(src_cx,dim2)

        idst_lo = cx_lo_s1(cx_lims)
        idst_hi = cx_hi_s1(cx_lims)
        jdst_lo = cx_lo_s2(cx_lims)
        jdst_hi = cx_hi_s2(cx_lims)

* flag for bad/missing data
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* DETERMINE (LOWER) src_grd INDEX OF NEAREST NEIGHBORS ON src_grd AXES
* FOR NODES ON dst_grd AXES, AND COMPUTE FRACTIONAL POSITIONS OF dst_grd
* NODES WITH RESPECT TO src_grd.
	CALL GET_LINEAR_COEF( isrc_lo, isrc_hi, src_grid,
     .			   idst_lo, idst_hi, dst_grid,
     .			   dim1, iss12, coefi )
	CALL GET_LINEAR_COEF( jsrc_lo, jsrc_hi, src_grid,
     .			   jdst_lo, jdst_hi, dst_grid,
     .			   dim2, jss12, coefj)

C LINEAR INTERPOLATION IN Y THEN X

* loop over Z, T, E, and F ranges of 6-D data
	DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* first in J direction.

	   DO j = jdst_lo,jdst_hi

* interpolation coefficients
	      jss = jss12(j)
	      c2j = coefj(j)
	      c1j = 1.0 - c2j

* special case: exact overlap of source and dest points
	      IF ( TM_FPEQ(c1j,1.) ) THEN
	         DO i = isrc_lo, isrc_hi
		    work(i,j) = src(i,jss,k,l,m,n)
	         ENDDO
		 CYCLE
	      ENDIF

* regular interpolation
	      DO i = isrc_lo, isrc_hi

* surrounding source data points
	         z1 = src(i,jss  ,k,l,m,n)
	         z2 = src(i,jss+1,k,l,m,n)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    work(i,j) = bad_dst
	         ELSE
	            work(i,j) = c1j*z1 + c2j*z2
	         ENDIF

	      ENDDO
	   ENDDO

* Now interpolate that in the x direction.

	   DO i = idst_lo, idst_hi

* interpolation coefficients
	      iss = iss12(i)
	      c2i = coefi(i)
	      c1i = 1.0 - c2i

* special case: exact overlap of source and dest points
	      IF ( TM_FPEQ(c1i,1.) ) THEN
	         DO j = jdst_lo, jdst_hi
		    dst(i,j,k,l,m,n) = work(iss,j)
	         ENDDO
		 CYCLE
	      ENDIF

* regular interpolation
	      DO j = jdst_lo, jdst_hi

* surrounding source data points
	         z1 = work(iss  ,j)
	         z2 = work(iss+1,j)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l,m,n) = bad_dst
	         ELSE
	            dst(i,j,k,l,m,n) = c1i*z1 + c2i*z2
	         ENDIF

	      ENDDO
	   ENDDO

	ENDDO  ! Close loops over Z, T, E, and F
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
