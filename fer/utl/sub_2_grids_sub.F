	SUBROUTINE SUB_2_GRIDS_SUB (
     .		plus, plox, phix, ploy, phiy, ploz, phiz, plot, phit, bad_plus,
     .		mnus, mlox, mhix, mloy, mhiy, mloz, mhiz, mlot, mhit, bad_mnus,
     .		rslt, rlox, rhix, rloy, rhiy, rloz, rhiz, rlot, rhit, bad_rslt )

* subtract 2 grids producing a third

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/12/89

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	plox, phix, ploy, phiy, ploz, phiz, plot, phit,
     .		mlox, mhix, mloy, mhiy, mloz, mhiz, mlot, mhit,
     .		rlox, rhix, rloy, rhiy, rloz, rhiz, rlot, rhit
	REAL    plus( plox:phix,ploy:phiy,ploz:phiz,plot:phit ),
     .		mnus( mlox:mhix,mloy:mhiy,mloz:mhiz,mlot:mhit ),
     .		rslt( rlox:rhix,rloy:rhiy,rloz:rhiz,rlot:rhit )
	REAL	bad_plus, bad_mnus, bad_rslt

* internal variable declarations:
	INTEGER	i, j, k, l

* --- end of introductory code ---

	DO 100 l = rlot, rhit
	DO 100 k = rloz, rhiz
	DO 100 j = rloy, rhiy
	DO 100 i = rlox, rhix

	   IF (  plus(i,j,k,l) .NE. bad_plus
     .	   .AND. mnus(i,j,k,l) .NE. bad_mnus ) THEN

	      rslt(i,j,k,l) = plus(i,j,k,l) - mnus(i,j,k,l)

	   ELSE

	      rslt(i,j,k,l) = bad_rslt

	   ENDIF

 100	CONTINUE

	RETURN
	END 
