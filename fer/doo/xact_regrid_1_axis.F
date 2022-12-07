	SUBROUTINE XACT_REGRID_1_AXIS(	cx_lims, axis, dst_lo, dst_hi,
     .					src, msrc, dst, mdst,
     .					ss12, coef )

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
*       Regrid from the source to the destination grid using an algorithm
*       of inserting values into the result ONLY when the source grid has
*       a value at exactly the same coordinate on this axis.

* 	msrc and mdst point to data structures in COMMON/XVARIABLES/ which
* 	specify the subscript bounds of src and dst within their respective
* 	grids.

* 	Both src and dst are assumed to be 4 dimensional structures
*	with regridding needed along axis "axis"

*	ss12 holds the indices of grid nodes on src_grd for which there 
*       is an exact match on dst_grd ... or unspecified_int4 if no match

* note that (for convenience) this routine reuses code from LIN_REGRID_1_AXIS
* using the values of the linear coefficients computed as the test of whether
* the source and destination grids have an exact match

* NOAA/PMEL,Seattle,WA - Tropical Modeling and Analysis Program

* V500: 10/97

* v570 *acm* 5/04 Range of subscripts might be negative, and contain 
*                 -111 among them. Allow regridding, using the flag 
*                 ok_neg111 to distinguish this case from the flag 
*                 unspecified_int4.
* v580 *acm* 11/04 Fix the definition of ok_neg111; if -111 is at
*                  either end of the interval we cannot distinguish
*                  it as a valid index from its use as the bad flag.
*        2/12 *kms* Add E and F dimensions (use nferdims in tmap_dims.parm)
* v744 *acm* 11/18 Issue 1238: For XACT regridding call customized routine 
*                  to compute regridding subscripts, coefficients	
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax.          

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xvariables.cmn'
	include	'xmem_subsc.cmn'
	include 'xcontext.cmn'


* CALLING ARGUMENT DECLARATIONS:
	INTEGER	cx_lims, axis, dst_lo, dst_hi, msrc, mdst
	INTEGER ss12( dst_lo:dst_hi )
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,
     .               m1lot:m1hit,m1loe:m1hie,m1lof:m1hif ),
     .          dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,
     .               m2lot:m2hit,m2loe:m2hie,m2lof:m2hif ),
     .          coef( dst_lo:dst_hi )

* INTERNAL VARIABLE DECLARATIONS:
	LOGICAL	TM_FPEQ_SNGL, TM_DFPEQ_TOL, ok_neg111
	INTEGER i, j, k, l, m, n,
     .		ss, src_lo, src_hi, src_grd, dst_grd, was
	REAL	bad_src, bad_dst
	REAL*4  singlec, singlez, single1

******************************************************************************

* LIMITS FOR CALCULATION
        src_lo = mr_lo_ss(msrc,axis)
        src_hi = mr_hi_ss(msrc,axis)

* FLAG FOR BAD/MISSING DATA
        bad_src = mr_bad_data(msrc)
        bad_dst = mr_bad_data(mdst)

* source and destination grids
	src_grd = mr_grid( msrc )
	dst_grd = mr_grid( mdst )

* single-precision comparison with zero and 1.
	singlez = 0.0  
	single1 = 1.0  

* DETERMINE src_grd INDEX OF exact matches on the dst_grd axis
	CALL GET_COEF_FOR_XACT(	src_lo, src_hi, src_grd,
     .				dst_lo, dst_hi, dst_grd,
     .				axis, ss12, coef )
	DO i = dst_lo, dst_hi
	   singlec = coef(i)
	   was = ss12(i)
	   IF ( .NOT.TM_FPEQ_SNGL(singlec,singlez) ) 
     .				ss12(i) = unspecified_int4
	   IF (  TM_FPEQ_SNGL(singlec,single1) ) 
     .				ss12(i) = was
	ENDDO

* Range of subscripts might be negative, and contain -111 among them.
* ok_neg111 distinguishes between this and the flag unspecified_int4.

        ok_neg111 = (ss12(dst_lo) .LT. unspecified_int4 .AND.
     .               ss12(dst_hi) .GT. unspecified_int4) .OR.
     .              (ss12(dst_lo) .GT. unspecified_int4 .AND.
     .               ss12(dst_hi) .LT. unspecified_int4) 

* transfer THE map-able DATA TO dst_grd
	IF ( axis .EQ. x_dim ) THEN
	   DO i = dst_lo, dst_hi

	      ss = ss12(i)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	            dst(i,j,k,l,m,n) = src(ss,j,k,l,m,n)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
		    dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO j = dst_lo, dst_hi

	      ss = ss12(j)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	            dst(i,j,k,l,m,n) = src(i,ss,k,l,m,n)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
		 dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO k = dst_lo, dst_hi

	      ss = ss12(k)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	            dst(i,j,k,l,m,n) = src(i,j,ss,l,m,n)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
		    dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO l = dst_lo, dst_hi

	      ss = ss12(l)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	            dst(i,j,k,l,m,n) = src(i,j,k,ss,m,n)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
		    dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO


	ELSEIF ( axis .EQ. e_dim ) THEN
	   DO m = dst_lo, dst_hi

	      ss = ss12(m)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	            dst(i,j,k,l,m,n) = src(i,j,k,l,ss,n)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO n = cx_lo_s6(cx_lims), cx_hi_s6(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
		    dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO


	ELSEIF ( axis .EQ. f_dim ) THEN
	   DO n = dst_lo, dst_hi

	      ss = ss12(n)

	      IF ( ss .NE. unspecified_int4 .OR. 
     .             ss .EQ. unspecified_int4 .AND. ok_neg111) THEN
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
	            dst(i,j,k,l,m,n) = src(i,j,k,l,m,ss)
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ELSE
	         DO m = cx_lo_s5(cx_lims), cx_hi_s5(cx_lims)
	         DO l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
		    dst(i,j,k,l,m,n) = bad_dst
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	         ENDDO
	      ENDIF

	   ENDDO

	ENDIF

	RETURN
	END
