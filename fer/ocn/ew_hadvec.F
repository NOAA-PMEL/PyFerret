	SUBROUTINE EW_HADVEC ( tk, mtk, uk, muk, advec, madvec )

* compute east-west component of heat advection

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/17/86	extracted from SUBROUTINE HDIFFUS
*			with major modifications to convert i-k oriented
*			calculations to i-j oriented calculations
* V200:  7/27/89 - 4D symmetrical version
*	10/11/89 - array declarations using XMEM_SUBSC.CMN (reordered args)

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
	INTEGER	muk, mtk, madvec
* subscript ranges from memory variable table ...
	REAL       tk( m1lox:m1hix,m1loy:m1hiy,m1loz:m1hiz,m1lot:m1hit ),
     .		   uk( m2lox:m2hix,m2loy:m2hiy,m2loz:m2hiz,m2lot:m2hit ),
     .		advec( m3lox:m3hix,m3loy:m3hiy,m3loz:m3hiz,m3lot:m3hit )
	REAL fuw(2:imtp1)

* internal variable declarations:
	INTEGER	i, j, k, l
	REAL	bad_tk, bad_uk, bad_qadvec, fact1, dutdxm, tdudx

* --- end of introductory code ---
* flag for bad/missing data
	bad_tk     = mr_bad_data( mtk )
	bad_uk     = mr_bad_data( muk )
	bad_qadvec = mr_bad_data( madvec )

* south to north loop through east-west lines
	DO 2000 l = mr_lo_s4(madvec), mr_hi_s4(madvec)
	DO 2000 k = mr_lo_s3(madvec), mr_hi_s3(madvec)
	DO 1000 j = mr_lo_s2(madvec), mr_hi_s2(madvec)

	IF ( j .EQ. 1 ) THEN
	   DO 100 i = mr_lo_s1(madvec), mr_hi_s1(madvec)
 100	   advec( i,j,k,l ) = bad_qadvec
	   GOTO 1000
	ENDIF

C   FIND ADVECTIVE COEFFICIENTS FOR WEST AND EAST FACES OF T,S BOXES	! 2967

      FACT1=CSTR(j)*DYTR(j)	    				! 2969 mod
      DO 690 i = MAX(2,mr_lo_s1(madvec)), mr_hi_s1(madvec)+1	! 2972 mod
	IF ( uk( i-1,j  ,k,l ) .EQ. bad_uk
     .  .OR. uk( i-1,j-1,k,l ) .EQ. bad_uk ) THEN
	   fuw(i) = bad_val4
	ELSE
           FUW(I)=( Uk(i-1,j,k,l)  *DYU(j)
     .		 +  Uk(I-1,j-1,k,l)*DYU(j-1) )*FACT1	! 2973 mod
	ENDIF
 690  CONTINUE	    							! 2976

* calculate minus east-west advection component of total advection in flux form
      DO 820 i = mr_lo_s1(madvec), mr_hi_s1(madvec)		! 3066 mod

	IF ( i .EQ. 1  .OR. i .EQ. im ) THEN
	   advec( i,j,k,l ) = bad_qadvec
	   GOTO 820
	ENDIF
	IF ( fuw( i   )    .EQ. bad_val4
     .	.OR. fuw( i+1 )    .EQ. bad_val4
     .	.OR. tk( i  ,j  ,k,l ) .EQ. bad_tk
     .	.OR. tk( i-1,j  ,k,l ) .EQ. bad_tk
     .	.OR. tk( i  ,j-1,k,l ) .EQ. bad_tk
     .	.OR. tk( i-1,j-1,k,l ) .EQ. bad_tk ) THEN
	   advec( i,j,k,l ) = bad_qadvec
	   GOTO 820
	ENDIF

	dutdxm =  (
     *          (-FUW(I+1)*(tk(i+1,j,k,l) + tk(i  ,j,k,l))    	      ! 3067 mod
     *           +FUW(I  )*(tk(i  ,j,k,l) + tk(i-1,j,k,l)))*DXT4R(I) )! 3068 mod

* calculate individual advection component by subtracting the vector t*dU/dX 
	tdudx = tk(i,j,k,l) * (fuw(i+1) - fuw (i))*dxt4r(i)*2.0
	advec(i,j,k,l) = (dutdxm + tdudx ) * cmonthly

 820  CONTINUE	    							! 3072

 1000	CONTINUE
 2000	CONTINUE

	RETURN
	END
