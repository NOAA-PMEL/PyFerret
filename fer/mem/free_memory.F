	SUBROUTINE FREE_MEMORY ( mblk, num_blocks )

* return a contiguous region of bulk memory to the free list merging it with
* the surrounding free regions if they are contiguous
* ( return the merged size via COMMON in mr_del_last_size )
* see documentation in XVARIABLES.CMN

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/1/86
* revision 1.0 - 7/21/88 - complete re-write using linked list
* V200:  6/12/89 - use doubly linked list; return mr_del_last_*
*	 8/1/89  - corrected bug: mr_del_last* not set for not contig at all
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'tmap_format/tmap_dims.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	mblk, num_blocks, mem_blk, i

* internal variable declarations:
	INTEGER	prev, next, nex2
	LOGICAL	contig_above, contig_below

! temp diag
	IF ( num_blocks .NE. mblk_size( mblk ) ) STOP ' FREE_MEMORY'

!* as a diagnostic test - fill unused memory with a flag
! removed 5/18/94 as "memory" is no longer in COMMON 
!	IF ( mode_diagnostic ) THEN
!	   DO 50 mem_blk = mblk, mblk+num_blocks-1
!	   DO 50 i = 1, mem_blk_size
! 50	   memory( i,mem_blk ) = bad_val4
!	ENDIF

* find the position of "mblk" in the ordered free memory list
	prev = 0
	next = mblk_flink( prev )
 100	IF ( next .EQ. 0 ) THEN
* ... end of list - this mblk value must be greater than anything in the list
	   GOTO 200
	ELSEIF ( next .LT. mblk ) THEN
* ... keep looking further in the list
	   prev = next
	   next = mblk_flink( next )
	   GOTO 100
	ENDIF
* ... found it - prev<mblk and next>mblk
	nex2 = mblk_flink( next )

* will the new region merge contiguously above or below ?
 200	contig_above = mblk + num_blocks .EQ. next
	contig_below = prev + mblk_size(prev) .EQ. mblk
     .		 .AND. prev .NE. 0

* insert the freed bulk memory region into the free list
	IF ( contig_above .AND. contig_below ) THEN
	   mr_del_last_mblk   = prev
	   mr_del_last_size   = mblk_size( prev )
     .			      + num_blocks
     .			      + mblk_size( next )
	   mblk_size ( prev ) = mr_del_last_size
	   mblk_flink( prev ) = nex2
	   mblk_blink( nex2 ) = prev
	   mblk_flink( mblk ) = unspecified_int4	! temp diag
	   mblk_blink( mblk ) = unspecified_int4	! temp diag
	   mblk_size ( mblk ) = unspecified_int4	! temp diag
	   mblk_flink( next ) = unspecified_int4	! temp diag
	   mblk_blink( next ) = unspecified_int4	! temp diag
	   mblk_size ( next ) = unspecified_int4	! temp diag

	ELSEIF ( contig_above ) THEN
	   mr_del_last_mblk   = mblk
	   mr_del_last_size   = num_blocks + mblk_size( next )
	   mblk_size ( mblk ) = mr_del_last_size
	   mblk_flink( mblk ) = nex2
	   mblk_blink( mblk ) = prev
	   mblk_blink( nex2 ) = mblk
	   mblk_flink( prev ) = mblk
	   mblk_flink( next ) = unspecified_int4	! temp diag
	   mblk_blink( next ) = unspecified_int4	! temp diag
	   mblk_size ( next ) = unspecified_int4	! temp diag

	ELSEIF ( contig_below ) THEN
	   mr_del_last_mblk   = prev
	   mr_del_last_size   = mblk_size( prev ) + num_blocks
	   mblk_size ( prev ) = mr_del_last_size
	   mblk_flink( mblk ) = unspecified_int4	! temp diag
	   mblk_blink( mblk ) = unspecified_int4	! temp diag
	   mblk_size ( mblk ) = unspecified_int4	! temp diag

	ELSE
	   mr_del_last_mblk   = mblk
	   mr_del_last_size   = num_blocks
	   mblk_flink( mblk ) = next
	   mblk_blink( mblk ) = prev
	   mblk_flink( prev ) = mblk
	   mblk_blink( next ) = mblk

	ENDIF
	RETURN
	END
