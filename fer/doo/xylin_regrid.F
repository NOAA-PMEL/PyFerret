	SUBROUTINE XYLIN_REGRID( cx_lims,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, src_cx, dst_cx, 
     .			         iss12, coefi, jss12, coefj, work)

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


	include 'ferret.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, msrc, mdst, dst_lox, dst_hix, dst_loy, dst_hiy,
     .          src_cx, dst_cx 
	INTEGER	iss12(dst_lox:dst_hix+1), jss12(dst_loy:dst_hiy+1)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		coefi( dst_lox:dst_hix ), coefj(dst_loy:dst_hiy)

* local variable declarations:

        LOGICAL		TM_FPEQ, do_bilinear
	INTEGER		dim1, dim2, src_grid, dst_grid,
     .                  i, j, k, l, 
     .                  isrc_lo, isrc_hi, jsrc_lo, jsrc_hi,
     .			idst_lo ,idst_hi, jdst_lo, jdst_hi, iss, jss
	REAL		bad_src, bad_dst, c1i, c1j, c2i, c2j, 
     .                  z1, z2, z3, z4
        REAL work(m1lox:m1hix,dst_loy:dst_hiy)  ! srcx by dsty for non-bilinear method
********************************************************************************

* initialize

	do_bilinear = .FALSE.

* source and destination grids
	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )
	dim1     = x_dim
	dim2     = y_dim

* limits for calculation
        isrc_lo = cx_lo_ss(src_cx,dim1)
        isrc_hi = cx_hi_ss(src_cx,dim1)
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

C BILINEAR INTERPOLATION IN XY.  Returns somewhat fewer points if theres missing data
C relative to all axes done separately, or if we do the various "interpolate in i,
C interpolate in j", fills in more than previously.
    
	IF (do_bilinear) THEN
* loop over Z and T ranges of 4-D data
	DO 70 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO 70 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* loop through the each destination grid box
	DO 60 j = jdst_lo,jdst_hi

* interpolation coefficients
	   jss = jss12(j)
	   c2j = coefj(j)
	   c1j = 1.0 - c2j
	   DO 50 i = idst_lo, idst_hi

	      dst(i,j,k,l) = bad_dst
	              a = dst(i,j,k,l)

* interpolation coefficients
	      iss = iss12(i)
	      c2i = coefi(i)
	      c1i = 1.0 - c2i
	   
* 4 surrounding source data points
	      z1 = src(iss  ,jss  ,k,l)
	      z2 = src(iss+1,jss  ,k,l)
	      z3 = src(iss+1,jss+1,k,l)
	      z4 = src(iss  ,jss+1,k,l)

* special cases: exact overlap of source and dest points
* This takes care of edges of the dest domain

	      IF ( TM_FPEQ(c1j,1.) .AND. TM_FPEQ(c1i,1.)) THEN
                  dst(i,j,k,l) = src(iss,jss,k,l)

	      ELSEIF ( (TM_FPEQ(c1j,1.) .AND. .NOT.TM_FPEQ(c1i,1.)) .AND.
     .                 (z3.EQ.bad_src .AND. z4.EQ.bad_src) ) THEN  ! interpolate just in i at this j

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1i*z1 + c2i*z2
	              a = dst(i,j,k,l)
	         ENDIF

	      ELSEIF ( (TM_FPEQ(c1j,1.) .AND. .NOT.TM_FPEQ(c1i,1.)) .AND.
     .                 (z1.EQ.bad_src .AND. z2.EQ.bad_src) ) THEN  ! interpolate just in i at next j

* interpolate
	         IF (z3 .EQ. bad_src .OR. z4 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1i*z3 + c2i*z4
	              a = dst(i,j,k,l)
	         ENDIF

	      ELSEIF ( (TM_FPEQ(c1i,1.) .AND. .NOT.TM_FPEQ(c1j,1.)) .AND.
     .                 (z2.EQ.bad_src .AND. z3.EQ.bad_src) ) THEN  ! interpolate just in j at this i

* interpolate
	         IF (z1 .EQ. bad_src .OR. z4 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1j*z1 + c2j*z4
	              a = dst(i,j,k,l)
	         ENDIF

	      ELSEIF ( (TM_FPEQ(c1i,1.) .AND. .NOT.TM_FPEQ(c1j,1.)) .AND.
     .                 (z1.EQ.bad_src .AND. z4.EQ.bad_src) ) THEN  ! interpolate just in j at next i

* interpolate
	         IF (z2 .EQ. bad_src .OR. z3 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1j*z2 + c2j*z3
	              a = dst(i,j,k,l)
	         ENDIF

	      ELSE

* interpolate

* interpolating just in j when points missing for interpolation in i direction
* mimics our previous behavior.  Uncomment the "interp just in i" lines to
* get more filling in.

	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src .OR.
     .                 z3 .EQ. bad_src .OR. z4 .EQ. bad_src) THEN
                    IF (z1 .NE. bad_src .AND. z4 .NE. bad_src) THEN ! interp just in j
	               dst(i,j,k,l) = c1j*z1 + c2j*z4
		       	              a = dst(i,j,k,l)

                    ELSEIF (z2 .NE. bad_src .AND. z3 .NE. bad_src) THEN ! interp just in j
	               dst(i,j,k,l) = c1j*z2 + c2j*z3
                    ELSEIF (z1 .NE. bad_src .AND. z2 .NE. bad_src) THEN ! interp just in i
	               dst(i,j,k,l) = c1i*z1 + c2i*z2
                    ELSEIF (z3 .NE. bad_src .AND. z4 .NE. bad_src) THEN ! interp just in j
	               dst(i,j,k,l) = c1i*z3 + c2i*z4
		    ELSE
	              dst(i,j,k,l) = bad_dst
	              a = dst(i,j,k,l)
		    ENDIF
	         ELSE
	            dst(i,j,k,l) = c1i*c1j*z1 + c2i*c1j*z2 + 
     .			           c2i*c2j*z3 + c1i*c2j*z4
	              a = dst(i,j,k,l)

	         ENDIF
              ENDIF

 50	   CONTINUE

 60	CONTINUE

 70	CONTINUE

	ELSE ! do_bilinear


C LINEAR INTERPOLATION IN Y THEN X
C This  replicates Ferrets previous behavior
    
* loop over Z and T ranges of 4-D data
	DO 300 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO 300 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* first in J direction.

	   DO 210 j = jdst_lo,jdst_hi

* interpolation coefficients
	      jss = jss12(j)
	      c2j = coefj(j)
	      c1j = 1.0 - c2j

* special case: exact overlap of source and dest points
	      IF ( TM_FPEQ(c1j,1.) ) THEN
	         DO 200 i = isrc_lo, isrc_hi
 200	            work(i,j) = src(i,jss,k,l)
	         GOTO 210
	      ENDIF

* regular interpolation
	      DO 205 i = isrc_lo, isrc_hi

* surrounding source data points
	         z1 = src(i,jss  ,k,l)
	         z2 = src(i,jss+1,k,l)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    work(i,j) = bad_dst
	         ELSE
	            work(i,j) = c1j*z1 + c2j*z2
	         ENDIF

 205	      CONTINUE
 210	   CONTINUE

* Now interpolate that in the x direction.

	   DO 110 i = idst_lo, idst_hi

* interpolation coefficients
	      iss = iss12(i)
	      c2i = coefi(i)
	      c1i = 1.0 - c2i

* special case: exact overlap of source and dest points
	      IF ( TM_FPEQ(c1i,1.) ) THEN
	         DO 100 j = jdst_lo, jdst_hi
 100	         dst(i,j,k,l) = work(iss,j)
	         GOTO 110
	      ENDIF

* regular interpolation
	      DO 105 j = jdst_lo, jdst_hi

* surrounding source data points
	         z1 = work(iss  ,j)
	         z2 = work(iss+1,j)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src ) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1i*z1 + c2i*z2
		    a = dst(i,j,k,l)
	         ENDIF

 105	      CONTINUE
 110	   CONTINUE

 300	CONTINUE

 
	ENDIF ! do_bilinear

	RETURN
	END
