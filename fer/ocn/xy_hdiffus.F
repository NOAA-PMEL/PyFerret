	SUBROUTINE XY_HDIFFUS ( tk, mtk, xy_qdiffus, mxy_qdiffus )

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
* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/7/86	extracted from SUBROUTINE qdiffus
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* V200:  7/26/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'		! parameter definitions
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xonedim.cmn'	! geometric constants
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	mtk, mxy_qdiffus
* subscript ranges from memory variable table ...
	REAL            tk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		xy_qdiffus( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit )

* internal variable declarations:
	LOGICAL HOURLY_DATA
	INTEGER	i, j, k, l, dl, lp1
	REAL	bad_tk, bad_qdiffus, cctj, ddtj, ew_qdiffus, sn_qdiffus

* --- end of introductory code ---
* use 1 hour separation for exact results on hourly data
	IF ( HOURLY_DATA(mr_grid(mxy_qdiffus)) ) THEN
	   dl = 1
	ELSE
	   dl = 0
	ENDIF

* flag for bad/missing data
	bad_tk      = mr_bad_data( mtk )
	bad_qdiffus = mr_bad_data( mxy_qdiffus )

	DO 2000 l = mr_lo_s4(mxy_qdiffus)-dl, mr_hi_s4(mxy_qdiffus)-dl
	   lp1 = l + dl
	DO 2000 k = mr_lo_s3(mxy_qdiffus), mr_hi_s3(mxy_qdiffus)

	DO 1000 j = mr_lo_s2(mxy_qdiffus), mr_hi_s2(mxy_qdiffus)

	IF ( j .EQ. 1  .OR.  j .EQ. jmt ) THEN
	   DO 100 i = mr_lo_s1(mxy_qdiffus), mr_hi_s1(mxy_qdiffus)
 100	   xy_qdiffus( i,j,k,lp1 ) = bad_qdiffus
	   GOTO 1000
	ENDIF

C   FORM LATERAL DIFFUSION COEFFICIENTS					! 3045
C     VARIABLE HORIZONTAL DIFFUSION					! 3047
      CCTJ=AH*CS(j)*DYUR(j)*DYTR(j)*CSTR(j)*XJV(j)		! 3049 mod (j)
      DDTJ=AH*CS(j-1)*DYUR(j-1)*DYTR(j)*CSTR(j)*XJV(j-1)	! 3050 mod (j)

	DO 900 i = mr_lo_s1(mxy_qdiffus), mr_hi_s1(mxy_qdiffus)	! 3051 mod

	IF ( i .EQ. 1  .OR.  i .EQ. imt ) THEN
	   xy_qdiffus( i,j,k,lp1 ) = bad_qdiffus
	   GOTO 900
	ENDIF

	IF (	tk( i  ,j   ,k,l ) .EQ. bad_tk
     .	   .OR. tk( i+1,j   ,k,l ) .EQ. bad_tk
     .	   .OR. tk( i-1,j   ,k,l ) .EQ. bad_tk
     .	   .OR. tk( i  ,j+1 ,k,l ) .EQ. bad_tk
     .	   .OR. tk( i  ,j-1 ,k,l ) .EQ. bad_tk ) THEN
	   xy_qdiffus( i,j,k,lp1 ) = bad_qdiffus
	   GOTO 900
	ENDIF

      AATJ(I)=AH*DXUR(I)*DXTR(I)*CSTR(j)*CSTR(j)*XIV(I)*XJT(j)	  ! 3052 mod (j)
      BBTJ(I)=AH*DXUR(I-1)*DXTR(I)*CSTR(j)*CSTR(j)*XIV(I-1)*XJT(j)! 3053 mod (j)
  731 CONTINUE								! 3056

C   ADD IN DIFFUSION TERMS						! 3086
C									! 3087
* compute east-west diffussion
	ew_qdiffus = cmonthly * (
     * aatj(I)*(tk(I+1,j,k,l)-tk(i,j,k,l))			! 3092 mod
     *+bbtj(I)*(tk(I-1,j,k,l)-tk(i,j,k,l)) )			! 3093 mod
	xy_qdiffus(i,j,k,lp1) = ew_qdiffus

* compute south-north diffussion
	sn_qdiffus = cmonthly * (
     * cctj*xit(i)*(tk(i,j+1,k,l)-tk(i,j,k,l))				! 3102 mod
     *+ddtj*xit(i)*(tk(i,j-1,k,l)-tk(i,j,k,l))	)			! 3103 mod
	 xy_qdiffus(i,j,k,lp1) = xy_qdiffus(i,j,k,lp1) + sn_qdiffus

 900	CONTINUE
 1000	CONTINUE
 2000	CONTINUE

	RETURN
	END
