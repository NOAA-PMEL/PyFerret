	SUBROUTINE DELETE_UN_CACHED_MVARS

* clear (delete) the entire list of uncached memory variables
* (sanity check along the way)

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
* ...

* internal variable declarations:
	INTEGER	mr

* initialize linked list search
	mr =  mv_flink( pmv_nocache_pile )

* gobble up the linked list
 100	IF ( mr .EQ. pmv_nocache_pile ) THEN
	   RETURN
	ELSE
	   CALL DELETE_VARIABLE( mr )		! delete head of list
	   mr =  mv_flink( pmv_nocache_pile )	! new head of list
	   GOTO 100
	ENDIF
	   
	END





