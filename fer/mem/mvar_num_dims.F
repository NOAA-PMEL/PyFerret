	INTEGER FUNCTION mr_NUM_DIMS ( mr )

* determine the number of non-normal axes for this memory variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/11/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	mr

* internal variable declarations
	INTEGER	mr_DIM_LEN, idim, num

	num = 0	
	DO 100 idim = 1, 4
	   IF ( mr_DIM_LEN ( idim, mr ) .GT. 1 ) num = num + 1
 100	CONTINUE

	mr_NUM_DIMS = num

	RETURN
	END
