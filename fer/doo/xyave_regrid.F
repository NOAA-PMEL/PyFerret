	SUBROUTINE XYAVE_REGRID( cx_lims,
     .				 dst_lox, dst_hix, dst_loy, dst_hiy,
     .				 src, msrc, dst, mdst,
     .			         src_cx, dst_cx, i21,j21)

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

	include 'ferret.parm'
	include 'xcontext.cmn'
	include	'xmem_subsc.cmn'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	cx_lims, msrc, mdst, dst_lox, dst_hix, dst_loy, dst_hiy,
     .          src_cx, dst_cx 
	INTEGER	i21(dst_lox:dst_hix+1),
     .		j21(dst_loy:dst_hiy+1)

	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* local variable declarations:
	LOGICAL		GEOG_COS_FACTOR, need_cos, ok_negi111, ok_negj111

	INTEGER		idst,isrc,jdst,jsrc,dim1,dim2,src_grid,dst_grid,k,l
	INTEGER	     	isrc_lo,isrc_hi,jsrc_lo,jsrc_hi,
     .			idst_lo,idst_hi,jdst_lo,jdst_hi

	REAL		AXIS_DISTANCE, BOX_SIZE, xdist, ydist, area, sum, 
     .			darea, bad_src, bad_dst, cos_fact, bx, yc, yh_2,
     .                  pi, r_earth, area_earth
	
        REAL*8		TWORLD, xsrc_lo, xsrc_hi, ysrc_lo, ysrc_hi,
     .				xdst_lo, xdst_hi, ydst_lo, ydst_hi
********************************************************************************

* initialize

	src_grid = mr_grid( msrc )
	dst_grid = mr_grid( mdst )
	dim1     = x_dim
	dim2     = y_dim

* need to include COSINE factor ?
	need_cos = GEOG_COS_FACTOR( dim2, dst_grid )

* limits for calculation
c        isrc_lo = mr_lo_ss(msrc,dim1)
c        isrc_hi = mr_hi_ss(msrc,dim1)
c        jsrc_lo = mr_lo_ss(msrc,dim2)
c        jsrc_hi = mr_hi_ss(msrc,dim2)


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

        ok_negi111 = ( (i21(idst_lo) .LE. unspecified_int4 .AND.
     .                 i21(idst_hi) .GE. unspecified_int4) ) .AND.
     .                (i21(idst_lo) .NE. i21(idst_hi) )

        ok_negj111 = ( (j21(jdst_lo) .LE. unspecified_int4 .AND.
     .                 j21(jdst_hi) .GE. unspecified_int4) ) .AND.
     .                (j21(jdst_lo) .NE. j21(jdst_hi) )


* loop over Z and T ranges of 4-D data
	DO 100 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	DO 100 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)

* loop through the each destination grid box
	DO 60 jdst = jdst_lo,jdst_hi
	jsrc_lo = j21(jdst)
	jsrc_hi = j21(jdst+1)
	ydst_lo = TWORLD(jdst, dst_grid, dim2, box_lo_lim )
	ydst_hi = TWORLD(jdst, dst_grid, dim2, box_hi_lim )
	DO 60 idst = idst_lo,idst_hi
	   isrc_lo = i21(idst)
	   isrc_hi = i21(idst+1)

*  check if requested region is out of source range
           IF ( ((isrc_lo .EQ. unspecified_int4
     .	  .OR.  isrc_hi .EQ. unspecified_int4) .AND. .NOT.ok_negi111)
     .	  .OR.  ((jsrc_lo .EQ. unspecified_int4
     .	  .OR.  jsrc_hi .EQ. unspecified_int4) .AND. .NOT.ok_negj111) ) 
     .     THEN
	      dst(idst,jdst,k,l) = bad_dst
	      goto 60
	   ENDIF

	   xdst_lo = TWORLD(idst, dst_grid, dim1, box_lo_lim )
	   xdst_hi = TWORLD(idst, dst_grid, dim1, box_hi_lim )


           pi = 3.141592653
           r_earth = 6371000
           area_earth = 4.* pi * r_earth**2

* add all partial or complete boxes of data from the source grid
* 3/06 - apply Andrew*s suggested correction, bug1348

	   sum  = 0.0
	   area = 0.0
	   DO 51 jsrc = jsrc_lo, jsrc_hi
	   cos_fact = 1.0
	   IF ( need_cos ) THEN
              cos_fact = COS(deg2rad*TWORLD(jsrc,src_grid,dim2,box_middle)) !old method!
              yc = TWORLD(jsrc,src_grid,dim2,box_middle)
              bx = BOX_SIZE( jsrc, src_grid, dim2 )

	      yh_2 = 0.5*deg2rad * bx
              yc = deg2rad * yc
              cos_fact = COS(yc) * 2.*ABS( SIN(yh_2) )
	   ENDIF

	   ysrc_lo = MAX( TWORLD(jsrc,src_grid,dim2,box_lo_lim), ydst_lo )
	   ysrc_hi = MIN( TWORLD(jsrc,src_grid,dim2,box_hi_lim), ydst_hi )
	   ydist   = AXIS_DISTANCE( ysrc_lo, ysrc_hi, dim2, src_grid )


	   DO 50 isrc = isrc_lo, isrc_hi

* ... ignore missing source data
	      IF ( src( isrc, jsrc, k, l ) .EQ. bad_src ) GOTO 50

	      xsrc_lo = MAX( TWORLD(isrc,src_grid,dim1,box_lo_lim), xdst_lo )
	      xsrc_hi = MIN( TWORLD(isrc,src_grid,dim1,box_hi_lim), xdst_hi )
	      xdist   = AXIS_DISTANCE( xsrc_lo, xsrc_hi, dim1, src_grid )

* ... contribution from this source box
              IF (xdist .gt. 0.  .AND.  ydist .GT. 0) THEN
 	         darea = ydist*(cos_fact*xdist)/area_earth
	         area  = area + darea
	         sum   = sum  + darea * src(isrc, jsrc, k, l)
              ENDIF

 50	      CONTINUE
 51	   CONTINUE
* ... divide by area to get average
	   IF ( area .GT. 0.0 ) THEN
	      dst(idst, jdst, k, l) = sum / area
c!c	      dst(idst, jdst, k, l) =  area
	   ELSE
	      dst(idst, jdst, k, l) = bad_dst
	   ENDIF

 60	CONTINUE

 100	CONTINUE
c!c        print *, ' hack in xyave_regrid: return area, not sum-of-var/area'

	RETURN
	END
