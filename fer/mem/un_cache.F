	SUBROUTINE UN_CACHE( mr )

* flag a memory variable as not cache-retreivable
* do this by shifting the variable to a "hash pile" which is not searched

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
	INTEGER	prev, next, pile, head

* remove variable slot from hash pile doubly linked list
	prev = mv_blink( mr )
	next = mv_flink( mr )
	mv_flink( prev ) = next
	mv_blink( next ) = prev

* add the memory variable to the head of the non-retreivable hash pile
	pile = pmv_nocache_pile
	head = mv_flink( pile )
	mv_flink( pile ) = mr
	mv_flink( mr ) = head
	mv_blink( mr ) = pile
	mv_blink( head ) = mr

	RETURN
	END


