      REAL FUNCTION RANDN(jd)

* compute a quasi-random number with a standard normal distribution
* based on the routine "GASDEV" on p.203 of Numerical Recipes
* W.H.Press,B.P.Flannery,S.A.Teukolsky,W.T.Vetterling

* uses routine RANDU to supply normal deviates

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:  3/29/90

* calling argument declarations:
	INTEGER	jd

* internal variable declarations:
	LOGICAL saved
	REAL	RANDU, v1, v2, r, fac, rsave

* initialize the uniform and this random generator
	IF ( jd.NE.0 ) THEN
	   v1 = RANDU(jd)
	   saved = .FALSE.
	ENDIF

	IF ( saved ) THEN
* ... use the one saved last time
	   RANDN = rsave
	   saved = .FALSE.

	ELSE
* ... compute 2 normal from 2 uniforms, return 1, save 1
 10	   v1 = 2. * RANDU(0) - 1.
	   v2 = 2. * RANDU(0) - 1.
	   r  = v1**2 + v2**2
	   IF ( r .GE. 1 ) GOTO 10	! not inside unit circle ?
	   fac = SQRT(-2.*LOG(r)/r)	   
	   rsave = v1*fac
	   RANDN = v2*fac
	   saved = .TRUE.

	ENDIF

	RETURN
	END
