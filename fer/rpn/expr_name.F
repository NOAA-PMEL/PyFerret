	CHARACTER*8 FUNCTION EXPR_NAME( n )

* create a unique name for a user-defined variable
* "expr#1","expr#2",..."expr#9","expr#:","expr#,",etc. from ASCII table

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/23/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER		n

	EXPR_NAME = 'EX#'//CHAR(48+n)

	RETURN
	END
