	SUBROUTINE RE_ASSIGN_VARIABLE( mr, new_cx )

* re-assign the data which is currently cataloged in the memory variable
* table under mr to the context information stored in new_cx

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87
* V200:  6/20/89 - improved deletion priority scheme
*		 - variable is assumed NOT to be in deletion chain
*		 - use ASSIGN_MR_CX

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
	INTEGER		mr, new_cx

* internal variable declarations:
	INTEGER		HASH_CX, prev, next, pile, head

* flag the variable as "in progress"
	mr_protected(mr) = mr_in_progress

* make this mem var the property of the new context
	CALL ASSIGN_MR_CX( new_cx, mr )

* correct the links to put this in the right hash pile
*    ... remove variable slot from hash pile doubly linked list
	   prev = mv_blink( mr )
	   next = mv_flink( mr )
	   mv_flink( prev ) = next
	   mv_blink( next ) = prev
*    ... add the slot to the head of the correct hash pile
	   pile = HASH_CX( new_cx )
	   head = mv_flink( pile )
	   mv_flink( pile ) = mr
	   mv_flink( mr ) = head
	   mv_blink( mr ) = pile
	   mv_blink( head ) = mr

	RETURN
	END
