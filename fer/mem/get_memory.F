	SUBROUTINE GET_MEMORY ( num_words, mblk, num_blocks, status )

* find and reserve a section of contiguous memory sufficient to hold num_words 
* procedure:
* first check the list of free memory regions to see if there's one big enough
* if not then delete variables working from the tail of the priority chain
* when an adequate region is found pluck what it needed of it from the free list

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/31/86
* revision 0.1 - 7/21/88 - no special call to "CLAIM_MEMORY" since FIND_MEMORY
*			   claims it, too, now
* V200:  6/12/89 - delete priorities handled through linked lists
*		 - eliminated FIND_MEMORY
*	  4/5/90 - spruced up error message
* V230:  4/30/92 - added coaching to error message

#ifdef unix
	include 'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	num_words, mblk, num_blocks, status

* internal variable declarations:
	INTEGER	mr, next, prev, size, new_mblk, num_slen
	CHARACTER  LEFINT*12, num_str*12

* COMPUTE # BLOCKS NEEDED
	num_blocks = ( num_words + mem_blk_size - 1 ) / mem_blk_size

* LOOK FOR A MEMORY REGION ALREADY AVAILABLE
* ... use linked list into free memory regions
	mblk = mblk_flink( 0 )

* ... nothing more to check ?
 100	IF ( mblk .EQ. 0 ) GOTO 200

* ... another region to check out - is it big enough ?
	size = mblk_size ( mblk )
	IF ( size .LT. num_blocks ) THEN
	   mblk = mblk_flink( mblk )
	   GOTO 100
	ELSE
	   GOTO 500
	ENDIF
	   
* NOTHING BIG ENOUGH IS AVAILABLE AT PRESENT
* delete from tail of deletion priority chain looking for something big enough
 200	mr = mr_del_blink(0)
	IF ( mr .EQ. 0 ) GOTO 5100
	CALL DELETE_VARIABLE( mr )
	size = mr_del_last_size
	IF ( size .LT. num_blocks ) THEN
	   GOTO 200
	ELSE
	   mblk = mr_del_last_mblk
	ENDIF

* WE'VE GOT A REGION AT LEAST BIG ENOUGH
 500	prev = mblk_blink( mblk )
	next = mblk_flink( mblk )

	IF ( size .EQ. num_blocks ) THEN
* ... perfect size - remove it from the free list
	   mblk_flink( prev ) = next
	   mblk_blink( next ) = prev

	ELSE
* ... unnecessarily big - bite off only what we need
	   new_mblk = mblk + num_blocks
	   mblk_flink( new_mblk ) = next
	   mblk_blink( new_mblk ) = prev
	   mblk_size ( new_mblk ) = size - num_blocks
	   mblk_flink( prev )     = new_mblk
	   mblk_blink( next )     = new_mblk
	ENDIF

* success - set up the requested region and give it to 'em
	mblk_size ( mblk ) = num_blocks
	status = ferr_ok

	mblk_flink( mblk ) = unspecified_int4	! temp diag
	mblk_blink( mblk ) = unspecified_int4	! temp diag

 5000	RETURN
 5100	num_str = LEFINT(num_words,num_slen)
	CALL ERRMSG( ferr_insuff_memory,status,
     .	num_str(:num_slen)//' words were requested', *5101)
 5101   CALL WARN('Use SET and SHOW REGION to ensure that the region'
     .             //' is explicit;')
        CALL WARN('then reissue your command.')
        CALL WARN('The "Memory use" section of the FERRET '
     .            //'Users Guide has further tips.') 
        GOTO 5000
	END
