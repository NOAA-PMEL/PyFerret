	SUBROUTINE  VEL_DIF_Z (	cdz  , mcdz  ,
     .				u    , mu    ,
     .				ubwi , mubwi  ,
     .				udfz , mudfz    )

* compute vertical diffusion of velocity components
* add wind stress effects if at surface
* note: the identical code applies to both U and V

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -1/23/87 extracted from SUBROUTINE CLINIC
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* revision 0.1 - 6/25/87 - added wind stress contribution at surface
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - modified array declarations using XMEM_SUBSC.CMN
*	 3/13/90 - fixed bug at k.eq.km

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
	INTEGER	mcdz, mu, mubwi, mudfz

* subscript ranges from memory variable table ...
	REAL     cdz( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   u( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		ubwi( m3lox:m3hix,m3loy:m3hiy,            m3lot:m3hit ),
     .		udfz( m4lox:m4hix,m4loy:m4hiy,m4loz:m4hiz,m4lot:m4hit )

* internal variable declarations:
	LOGICAL HOURLY_DATA, need_wind
	INTEGER	i_lo, i_hi, j_lo, j_hi, i, j, k, l, dl, lp1
	REAL	bad_u, bad_cdz, bad_ubwi, up, down

* --- end of introductory code ---
* use 1 hour separation for exact results on hourly data
	IF ( HOURLY_DATA(mr_grid(mudfz)) ) THEN
	   dl = 1
	ELSE
	   dl = 0
	ENDIF

* limits for calculation
	i_lo = mr_lo_s1(mudfz)
	i_hi = mr_hi_s1(mudfz)
	j_lo = mr_lo_s2(mudfz)
	j_hi = mr_hi_s2(mudfz)

* flag for bad/missing data
	bad_u    = mr_bad_data( mu )
	bad_cdz  = mr_bad_data( mcdz )
	bad_ubwi = mr_bad_data( mubwi )

* ***************************************************************
* ORIGINAL GFDL 205 CODE UPON WHICH CALCULATION IS BASED
*
* 261       DO 280 K=1,KM
* 262       DO 280 I=1,IMT
* 263       UDIF(I,K)=UB(I,K)
* 264       VDIF(I,K)=VB(I,K)
* 265  280  CONTINUE

* 276       DO 295 I=1,IMT
* 277       KZ=KMU(I)
* 278       UDIF(I,KZ+1)=UB(I,KZ)
* 279       VDIF(I,KZ+1)=VB(I,KZ)
* 280  295  CONTINUE

* 352 C   ADD IN VERTICAL ADVECTION AND VERTICAL DIFFUSION

* 362       DO 345 K=1,KMP1
* 363       DO 345 I=1,IMT
* 364       TEMP2D(I,K)=UDIF(I,K-1)-UDIF(I,K)
* 365  345  CONTINUE
* 366       DO 348 K=1,KM
* 367       DO 348 I=1,IMT
* 368       UA(I,K)=UA(I,K)+EEM2D(I,K)*VVC(I,K-1)*TEMP2D(I,K)-
* 369      * FFM2D(I,K)*VVC(I,K)*TEMP2D(I,K+1)
* 370  348  CONTINUE

* 378       DO 355 K=1,KMP1
* 379       DO 355 I=1,IMT
* 380       TEMP2D(I,K)=VDIF(I,K-1)-VDIF(I,K)
* 381  355  CONTINUE
* 382       DO 356 K=1,KM
* 383       DO 356 I=1,IMT
* 384       VA(I,K)=VA(I,K)+EEM2D(I,K)*VVC(I,K-1)*TEMP2D(I,K)-
* 385      * FFM2D(I,K)*VVC(I,K)*TEMP2D(I,K+1)
* 386  356  CONTINUE
*************************************************************************
	DO 2000 l = mr_lo_s4(mudfz)-dl, mr_hi_s4(mudfz)-dl
	   lp1 = l + dl
	DO 2000 k = mr_lo_s3(mudfz), mr_hi_s3(mudfz)

* flag for surface wind effects
	need_wind = k .EQ. 1

	DO 345 j = j_lo, j_hi
	DO 345 i = i_lo, i_hi

	   IF (   u(i,j,k,l) .EQ. bad_u
     .	    .OR.cdz(i,j,k,l) .EQ. bad_cdz ) THEN
	       udfz(i,j,k,lp1) = bad_val4
	      GOTO 345
	   ENDIF

	   IF ( k .GT. 1 ) THEN
	      up = eem(k) * cdz(i,j,k-1,l) * ( u(i,j,k-1,l) - u(i,j,k,l) )
	   ELSE
	      up = 0.0
	   ENDIF

	   IF ( k .LT. km ) THEN
	      down = ffm(k) * cdz(i,j,k,l) * ( u(i,j,k,l) - u(i,j,k+1,l) )
	   ELSE
	      down = 0.0
	   ENDIF

	   udfz(i,j,k,lp1) = ( up - down ) * cmonthly

* add wind ?
	   IF ( need_wind ) THEN
	      IF ( ubwi(i,j,l) .EQ. bad_ubwi ) THEN
	         udfz(i,j,1,lp1) = bad_val4
	      ELSE
	         udfz(i,j,1,lp1) = udfz(i,j,1,lp1) + ubwi(i,j,l)
	      ENDIF
	   ENDIF

 345	CONTINUE
 2000	CONTINUE
	RETURN
	END
