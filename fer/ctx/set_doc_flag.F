	SUBROUTINE SET_DOC_FLAG( select, flags, init )

* maintain the document flag structure
* if "init" then preset all flags to false

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:	  2/2/90
* V314: 8/24/94 *kob* IBM port - variable flag should be declared as logical and
*				 not integer...caught by AIX xlf compiler
#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations
* move flags variable from integer to logical declaration *kob* 8/94
	LOGICAL	init, flags(0:10)
	INTEGER select

* internal variable declarations
	INTEGER	i

* initialize flags ?
	IF ( init ) THEN
	   DO 100 i = 0, 10
 100	   flags(i) = .FALSE.
	ENDIF

	IF ( select .EQ. 0 ) RETURN

* set individual flag
	flags( select ) = .TRUE.

* set flag(0) to indicate that any other flag has been set
	flags(0)    = .TRUE.

	RETURN
	END
