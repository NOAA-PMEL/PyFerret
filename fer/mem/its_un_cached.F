	LOGICAL FUNCTION ITS_UN_CACHED( mr )

* determine if a particular memory variable is not cache retreivable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	mr

* internal variable declarations:
	INTEGER	mr_test

* initialize linked list search
	mr_test =  mv_flink( pmv_nocache_pile )

* loop through the uncached vars to see if it is here
 100	IF ( mr_test .EQ. pmv_nocache_pile ) THEN
	   ITS_UN_CACHED = .FALSE.
	   RETURN
	ELSEIF ( mr .EQ. mr_test ) THEN
	   ITS_UN_CACHED = .TRUE.
	   RETURN
	ELSE
	   mr_test = mv_flink(mr_test)
	   GOTO 100
	ENDIF
	   
	END


