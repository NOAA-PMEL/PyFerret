	SUBROUTINE FIND_MR_SLOT( cx, mr, status )

* locate an unused slot in the memory variable table and assign it to the
* hash pile associated with the given context (cx)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/21/88
* V200:  6/12/89 - delete priorities handled through linked lists

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, mr, status

* internal variable declarations:
	INTEGER		HASH_cx, pile, head

* is there a slot in the free slot list ?
 100	IF ( mv_flink( 0 ) . EQ. 0 ) THEN
* ... no - delete lowest priority variable to make room and try again
	   mr = mr_del_blink(0)
	   IF ( mr .EQ. 0 ) GOTO 5010
	   CALL DELETE_VARIABLE( mr )
	ENDIF

* transfer the slot from the free list to the cx hash pile
*    ... remove from free list head
	mr = mv_flink( 0 )
	mv_flink( 0 ) = mv_flink( mr )
*    ... add to head of hash pile
	pile = HASH_cx( cx )
	head = mv_flink( pile )
	mv_flink( pile ) = mr
	mv_flink( mr ) = head
	mv_blink( mr ) = pile
	mv_blink( head ) = mr

* successful completion
	status = ferr_ok
 5000	RETURN

* error exit : no space available
 5010	CALL ERRMSG( ferr_too_many_vars, status,' ', *5000 )
	END
