	SUBROUTINE XACT_REGRID_1_AXIS(	cx_lims, axis, dst_lo, dst_hi,
     .					src, msrc, dst, mdst,
     .					ss12, coef )

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

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* CALLING ARGUMENT DECLARATIONS:
	INTEGER	  cx_lims, axis, dst_lo, dst_hi, msrc, mdst
	INTEGER   ss12(	dst_lo:dst_hi )
	REAL    src( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		dst( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .          coef( dst_lo:dst_hi )

* INTERNAL VARIABLE DECLARATIONS:
	LOGICAL	TM_FPEQ
	INTEGER i, j, k, l,
     .		ss, src_lo, src_hi, src_grd, dst_grd
	REAL	bad_src, bad_dst

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

* DETERMINE src_grd INDEX OF exact matches on the dst_grd axis
	CALL GET_LINEAR_COEF(	src_lo, src_hi, src_grd,
     .				dst_lo, dst_hi, dst_grd,
     .				axis, ss12, coef )
	DO 10 i = dst_lo, dst_hi
	   IF ( .NOT.TM_FPEQ(coef(i),0.0) ) ss12(i) = unspecified_int4
 10	CONTINUE

* transfer THE map-able DATA TO dst_grd
	IF ( axis .EQ. x_dim ) THEN
	   DO 110 i = dst_lo, dst_hi

	      ss = ss12(i)

	      IF ( ss .NE. unspecified_int4 ) THEN
	         DO 100 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 100 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 100 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
 100	         dst(i,j,k,l) = src(ss,j,k,l)
	      ELSE
	         DO 102 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 102 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 102 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
 102		 dst(i,j,k,l) = bad_dst
	      ENDIF

 110	   CONTINUE

	ELSEIF ( axis .EQ. y_dim ) THEN
	   DO 210 j = dst_lo, dst_hi

	      ss = ss12(j)

	      IF ( ss .NE. unspecified_int4 ) THEN
	         DO 200 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 200 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 200 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 200	         dst(i,j,k,l) = src(i,ss,k,l)
	      ELSE
	         DO 202 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 202 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 202 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 202		 dst(i,j,k,l) = bad_dst
	      ENDIF

 210	   CONTINUE

	ELSEIF ( axis .EQ. z_dim ) THEN
	   DO 310 k = dst_lo, dst_hi

	      ss = ss12(k)

	      IF ( ss .NE. unspecified_int4 ) THEN
	         DO 300 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 300 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 300 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 300	         dst(i,j,k,l) = src(i,j,ss,l)
	      ELSE
	         DO 302 l = cx_lo_s4(cx_lims), cx_hi_s4(cx_lims)
	         DO 302 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 302 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 302		 dst(i,j,k,l) = bad_dst
	      ENDIF

 310	   CONTINUE

	ELSEIF ( axis .EQ. t_dim ) THEN
	   DO 410 l = dst_lo, dst_hi

	      ss = ss12(l)

	      IF ( ss .NE. unspecified_int4 ) THEN
	         DO 400 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 400 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 400 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 400	         dst(i,j,k,l) = src(i,j,k,ss)
	      ELSE
	         DO 402 k = cx_lo_s3(cx_lims), cx_hi_s3(cx_lims)
	         DO 402 j = cx_lo_s2(cx_lims), cx_hi_s2(cx_lims)
	         DO 402 i = cx_lo_s1(cx_lims), cx_hi_s1(cx_lims)
 402		 dst(i,j,k,l) = bad_dst
	      ENDIF

 410	   CONTINUE
	ENDIF

	RETURN
	END
