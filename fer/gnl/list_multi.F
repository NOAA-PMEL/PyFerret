	SUBROUTINE LIST_MULTI( memory, mr_list, cx_list,
     .			       heading, perm, wsize, work, status )

* control routine to set up working arrays for LIST_MULTI_SUB

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/31/90
* V312: 5/94 - array "memory" as a calling argument

* calling argument declarations:
	LOGICAL	heading
	INTEGER	mr_list(*), cx_list(*), perm(4), wsize, status
	REAL	memory(*), work(0:wsize)
	
* internal variable declarations:
	INTEGER ws

* the work array is partitioned as follows in the called routine:
*	work(0)	     is xact_range(nvar)
*	work(ws)     is has_size(4,nvar)
*	work(5*ws)   is bad(nvar)
*	work(6*ws)   is pt_blk(nvar)
*	work(7*ws)   is size(4,nvar)
*	work(11*ws)  is msize(nvar)
*	work(12*ws)  is width(nvar)
*	work(13*ws)  is out(nvar)

	ws = wsize / 14
	CALL LIST_MULTI_SUB(memory,mr_list,cx_list,heading,perm, ws,
     .			work(0),     work(ws),   work(5*ws),
     .			work(6*ws),  work(7*ws), work(11*ws),
     .			work(12*ws), work(13*ws),
     .			status)


	RETURN
	END
