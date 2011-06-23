	SUBROUTINE XYLIN_REGRID( cx_lims,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst, src_cx, dst_cx, 
     .			         iss12, jss12, coefi, coefj)

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

* programmer - Ansley Manke, from xyave_regrid.F
* NOAA/PMEL, Seattle, WA - TMAP
*
* 6/22/2011

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

	INTEGER		dim1, dim2, src_grid, dst_grid,
     .                  i, j, k, l, 
     .                  isrc_lo, isrc_hi, jsrc_lo, jsrc_hi,
     .			idst_lo ,idst_hi, jdst_lo, jdst_hi
	REAL		bad_src, bad_dst, c1i, c1j, c2i, c2j, 
     .                  z1, z2, z3, z4, iss, jss
********************************************************************************

* initialize

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
    
* loop over Z and T ranges of 4-D data
	DO 100 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO 100 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* loop through the each destination grid box
	DO 60 j = jdst_lo,jdst_hi

* interpolation coefficients
	   jss = jss12(j)
	   c2j = coefj(j)
	   c1j = 1.0 - c2j
	   DO 50 i = idst_lo, idst_hi

* interpolation coefficients
	      iss = iss12(i)
	      c2i = coefi(i)
	      c1i = 1.0 - c2i

* surrounding source data points
	         z1 = src(iss  ,jss  ,k,l)
	         z2 = src(iss+1,jss  ,k,l)
	         z3 = src(iss+1,jss+1,k,l)
	         z4 = src(iss  ,jss+1,k,l)

* interpolate
	         IF (z1 .EQ. bad_src .OR. z2 .EQ. bad_src .OR.
     .               z3 .EQ. bad_src .OR. z4 .EQ. bad_src) THEN
		    dst(i,j,k,l) = bad_dst
	         ELSE
	            dst(i,j,k,l) = c1i*c1j*z1 + c2i*c1j*z2 + 
     .			           c2i*c2j*z3 + c1i*c2j*z4
	         ENDIF

 50	      CONTINUE

 60	CONTINUE

 100	CONTINUE

	RETURN
	END
