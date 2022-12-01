	SUBROUTINE XYAVE_REGRID( cx_lims,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst,
     .			         src_cx, dst_cx, i21,j21)

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
* 	Using data values from src organized on grid src_grid, regrid to
* 	grid dst_grid by averaging withing boxes

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which

* 	specify the subscript bounds of src and dst within their 
* 	respective grids.

* 	Both src and dst are assumed to be 2 dimensional structures
* 	lying in the same plane - given by mr_DIM1 and mr_DIM2.

*	I21 and J21 hold the indices of source grid boxes (on src_grid) for box
*	limits on dst_grid.

* 	programmer - Steve Hankin
* 	NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program
* 	written for VAX computer under VMS operating system

* 	revision 0.0 -  1/9/89 
* V200: 12/1/89 - 4D symmetrical version
* note: the conversion to 4D is of the Q&D variety.  Optimization is possible
*	by providing additional work space and pre-computing the box sizes
*	along the X and Y axes
* V314&V400: 9/94 - bug fix: GOTO 50 ==> GOTO 60 for missing source value

* v552: *acm* 4/03 In some cases the source indices include a range
*                  outside the destination indices, so xdist or ydist < 0. 
*                  Test for this and only have a contribution to the avg from 
*                  portions of the source axis that overlap the dest box.
*                  Also, use src_lo_ss and src_hi_ss for the range on the src
*                  axis rather than mr_lo_ss and mr_hi_ss
* v570 *acm* 5/04 Range of subscripts might be negative, and contain 
*                 -111 among them. Allow regridding, using the flags 
*                 ok_negi111, ok_negj111 to distinguish this case from 
*                 the flag unspecified_int4.
* V600  4/06 *acm* - make refinement to cos(lat) factor as suggested by A. Wittenberg
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* 6/06 - Fix to Andrew*s correction for XY integ., bug1348
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, msrc, mdst, dst_lox, dst_hix,
     .          dst_loy, dst_hiy, src_cx, dst_cx 
	INTEGER	i21(dst_lox:dst_hix+1),
     .		j21(dst_loy:dst_hiy+1)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif )

* local variable declarations:
	LOGICAL GEOG_COS_FACTOR,
     .          need_cos, ok_negi111, ok_negj111

	INTEGER idst, isrc, jdst, jsrc, dim1, dim2,
     .          src_grid, dst_grid, k, l, m, n,
     .          isrc_lo, isrc_hi, jsrc_lo, jsrc_hi,
     .          idst_lo, idst_hi, jdst_lo, jdst_hi

	REAL	AXIS_DISTANCE, BOX_SIZE,
     .          xdist, ydist, area, sum, darea, bad_src, bad_dst,
     .          cos_fact, area_earth, ybxlo, ybxhi, yh 

        REAL*8	TWORLD,
     .          xsrc_lo, xsrc_hi, ysrc_lo, ysrc_hi,
     .          xdst_lo, xdst_hi, ydst_lo, ydst_hi
********************************************************************************

* initialize

	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )
	dim1     = x_dim
	dim2     = y_dim

* need to include COSINE factor ?
	need_cos = GEOG_COS_FACTOR( dim2, dst_grid )

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

* determine the source grid boxes containing the destination box edges

	CALL GET_AVE_LIMS( isrc_lo, isrc_hi, src_grid,
     .			   idst_lo, idst_hi, dst_grid,
     .			   dim1, i21 )
	CALL GET_AVE_LIMS( jsrc_lo, jsrc_hi, src_grid,
     .			   jdst_lo, jdst_hi, dst_grid,
     .			   dim2, j21 )

    
* Range of subscripts might be negative, and contain -111 among them.
* ok_negi111, ok_negj111 distinguish between this and the flag unspecified_int4.

        ok_negi111 = ( i21(idst_lo) .LT. unspecified_int4 .AND.
     .                 i21(idst_hi) .GT. unspecified_int4 )

        ok_negj111 = ( j21(jdst_lo) .LT. unspecified_int4 .AND.
     .                 j21(jdst_hi) .GT. unspecified_int4 )


* loop over Z, T, E, and F ranges of 6-D data
	DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* loop through the each destination grid box
	DO jdst = jdst_lo,jdst_hi
	jsrc_lo = j21(jdst)
	jsrc_hi = j21(jdst+1)
	ydst_lo = TWORLD(jdst, dst_grid, dim2, box_lo_lim )
	ydst_hi = TWORLD(jdst, dst_grid, dim2, box_hi_lim )
	DO idst = idst_lo,idst_hi
	   isrc_lo = i21(idst)
	   isrc_hi = i21(idst+1)

*  check if requested region is out of source range
           IF ( ( (isrc_lo .EQ. unspecified_int4
     .    .OR.     isrc_hi .EQ. unspecified_int4)
     .            .AND. .NOT. ok_negi111 )
     .    .OR.  ( (jsrc_lo .EQ. unspecified_int4
     .    .OR.     jsrc_hi .EQ. unspecified_int4)
     .            .AND. .NOT. ok_negj111 ) ) THEN
	      dst(idst,jdst,k,l,m,n) = bad_dst
	      CYCLE
	   ENDIF

	   xdst_lo = TWORLD(idst, dst_grid, dim1, box_lo_lim )
	   xdst_hi = TWORLD(idst, dst_grid, dim1, box_hi_lim )

! m_radius in ferret.parm
! area_earth used as a scaling factor that gets cancelled out; is it really needed?
           area_earth = 4. * 3.141592653 * m_radius**2

* add all partial or complete boxes of data from the source grid
* 3/06 - apply Andrew*s suggested correction, bug1348

	   sum  = 0.0
	   area = 0.0
	   DO jsrc = jsrc_lo, jsrc_hi
	   cos_fact = 1.0
	   IF ( need_cos ) THEN

C See this thread:
C http://www.pmel.noaa.gov/maillists/tmap/ferret_users/fu_2009/msg00506.html
C And also ticket 1348.

              ybxlo = deg2rad* TWORLD(jsrc,src_grid,dim2,box_lo_lim)
              ybxhi = deg2rad* TWORLD(jsrc,src_grid,dim2,box_hi_lim)
	      
	      yh = ybxhi-ybxlo  
              cos_fact = ABS( SIN(ybxhi) - SIN(ybxlo) )/ yh

	   ENDIF

	   ysrc_lo = MAX( TWORLD(jsrc,src_grid,dim2,box_lo_lim), ydst_lo)
	   ysrc_hi = MIN( TWORLD(jsrc,src_grid,dim2,box_hi_lim), ydst_hi)
	   ydist   = AXIS_DISTANCE( ysrc_lo, ysrc_hi, dim2, src_grid )

	   DO isrc = isrc_lo, isrc_hi

* ... ignore missing source data
	      IF ( src(isrc,jsrc,k,l,m,n) .EQ. bad_src ) CYCLE

	      xsrc_lo = MAX( TWORLD(isrc,src_grid,dim1,box_lo_lim), xdst_lo )
	      xsrc_hi = MIN( TWORLD(isrc,src_grid,dim1,box_hi_lim), xdst_hi )
	      xdist   = AXIS_DISTANCE( xsrc_lo, xsrc_hi, dim1, src_grid )

* ... contribution from this source box
*     1/area_earth factor cancels out in result average
              IF (xdist .gt. 0.  .AND.  ydist .GT. 0) THEN
	         darea = ydist*(cos_fact*xdist) / area_earth
	         area  = area + darea
	         sum   = sum  + darea * src(isrc,jsrc,k,l,m,n)
              ENDIF

	      ENDDO ! isrc
	   ENDDO    ! jsrc

* ... divide by area to get average

	   IF ( area .GT. 0.0 ) THEN
	      dst(idst,jdst,k,l,m,n) = sum / area
	   ELSE
	      dst(idst,jdst,k,l,m,n) = bad_dst
	   ENDIF

	ENDDO  ! Close loops over idst, jdst
	ENDDO

	ENDDO  ! Close loops over Z, T, E, and F
	ENDDO
	ENDDO
	ENDDO

	RETURN
	END
