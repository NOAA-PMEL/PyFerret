	SUBROUTINE  VEL_ADV_Z (	uk  , muk  ,
     .				wk  , mwk  ,
     .				uadz, muadz )

* compute vertical advection of velocity components
* note: the identical code applies to both U and V

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/23/87 extracted from SUBROUTINE CLINIC
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 4/13/87 - corrected dz(1) --> dz(k)
*			   and corrected flux form to true z advection
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
* Linux Port 1/97 *kob* - Added a preprocessor include for xgfdl_masks.cmn
*			  because it needed a preprocessor.
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/gfdl.parm'		! parameter definitions
	include	'ferret_cmn/xvariables.cmn'
	include	'ferret_cmn/xmem_subsc.cmn'
	include 'ferret_cmn/xonedim.cmn'	! geometric constants
#	include "ferret_cmn/xgfdl_masks.cmn"
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:GFDL.PARM'		! parameter definitions
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XMEM_SUBSC.CMN'
	INCLUDE 'FERRET_CMN:XONEDIM.CMN'	! geometric constants
	INCLUDE 'FERRET_CMN:XGFDL_MASKS.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	muk, mwk, muadz

* subscript ranges from memory variable table ...
	REAL      uk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		  wk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		uadz( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )

* internal variable declarations:
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l
	REAL	bad_u, bad_w, factor, up, down, dwudz, udwdz

* --- end of introductory code ---

* limits for calculation
	i_lo = mr_lo_s1(muadz)
	i_hi = mr_hi_s1(muadz)
	j_lo = mr_lo_s2(muadz)
	j_hi = mr_hi_s2(muadz)

* flag for bad/missing data
	bad_u = mr_bad_data( muk )
	bad_w = mr_bad_data( mwk )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 352 C   ADD IN VERTICAL ADVECTION AND VERTICAL DIFFUSION
* 353 C
* 354       DO 340 K=1,KMP1
* 355       DO 340 I=1,IMT
* 356       TEMP2D(I,K)=W(I,K)*(U(I,K-1)+U(I,K))
* 357  340  CONTINUE
* 358       DO 343 K=1,KM
* 359       DO 343 I=1,IMT
* 360       UA(I,K)=UA(I,K)+(TEMP2D(I,K+1)-TEMP2D(I,K))*DZ2R2D(I,K)
* 361  343  CONTINUE

* 371       DO 350 K=1,KMP1
* 372       DO 350 I=1,IMT
* 373       TEMP2D(I,K)=W(I,K)*(V(I,K-1)+V(I,K))
* 374  350  CONTINUE
* 375       DO 353 K=1,KM
* 376       DO 353 I=1,IMT
* 377  353  VA(I,K)=VA(I,K)+(TEMP2D(I,K+1)-TEMP2D(I,K))*DZ2R2D(I,K)
*************************************************************************
	DO 2000 l = mr_lo_s4(muadz), mr_hi_s4(muadz)
	DO 2000 k = mr_lo_s3(muadz), mr_hi_s3(muadz)

* note: k subscripting is offset by one in this code as compared to the 205 code
*	because w(0) = 0. in this code whereas w(1)=0 in the 205 version

	factor = (1. / dz(k)) * cmonthly

	DO 340 j = j_lo, j_hi
	DO 340 i = i_lo, i_hi

	   IF ( uk(i,j,k,l) .EQ. bad_u
     .	    .OR.wk(i,j,k,l) .EQ. bad_w ) THEN

	      uadz(i,j,k,l) = bad_val4
	      GOTO 340
	   ENDIF

* calculate flux form ( as in 205 code ): d/dz(WU)
	   IF ( k .GT. 1 ) THEN
	      up = wk(i,j,k-1,l) * ( uk(i,j,k-1,l) + uk(i,j,k,l) ) / 2
	   ELSE
	      up = 0.0
	   ENDIF

	   IF ( k .LT. kmu(i,j) ) THEN
	      down = wk(i,j,k,l) * ( uk(i,j,k,l) + uk(i,j,k+1,l) ) / 2
	   ELSE
	      down = 0.0
	   ENDIF

	   dwudz = ( up - down ) * factor

* calculate U*d/dz(W)
	   IF ( k .EQ. 1 ) THEN
	      udwdz = uk(i,j,k,l) * (     0.0       - wk(i,j,k,l) ) * factor
	   ELSE
	      udwdz = uk(i,j,k,l) * ( wk(i,j,k-1,l) - wk(i,j,k,l) ) * factor
	   ENDIF

* note: actually we need to calculate the negative: -W*d/dz(U)
* subtract to get -W*d/dz(U) = U*d/dz(W) - d/dz(WU)
	   uadz(i,j,k,l) = udwdz - dwudz

 340	CONTINUE
 2000	CONTINUE
	RETURN
	END
