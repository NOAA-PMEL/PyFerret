	LOGICAL FUNCTION INVALID_R8_TO_I4 ( value )

* test to see if the given REAL*8 value is convertible to an INTEGER*4

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 10/17/86
* V230:  6/26/92 - changed vax_machine.parm to accomodate limitations
*                  of Macintosh Language Systems FORTRAN:  DBLE cannot be used
*                  in a parameter definition

#ifdef unix
	include	'ferret_cmn/vax_machine.parm'
#else
	INCLUDE	'FERRET_CMN:VAX_MACHINE.PARM'
#endif

	REAL*8	value

* max and min INTEGER*4 expressed as REAL*8
	REAL*8		test_neg
	PARAMETER     ( test_neg = -( r8_int4_max + 1 ) )

	INVALID_R8_TO_I4 = ( value .LT. test_neg )
     .                .OR. ( value .GT. r8_int4_max )

	RETURN
	END
	
