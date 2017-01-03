	SUBROUTINE SANITY_CHECK_MEMORY(tag)

*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administrations (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* diagnostic report on memory usage and integrity check

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702: *sh* 11/16 This is an expanded version of CHECK_MEMORY intended 
*      for use with -D debug copilations

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xwork_space.cmn'

* calling argument declarations:
	INTEGER	lun
	CHARACTER*(*) tag

* internal variable declarations:
	LOGICAL mv_accounted_for(max_mrs)
	INTEGER	mv, mb, mvfree, mvused, mvuncached, mvwork,
     .		mbfree, mbused, mbmax,
     .		nmvlost, nmblost, n, pile, last, size, test_next,
     .		ninuse, navail, nperm, mbperm, ntemp, ndel,
     .		ninprogress, ntable_entries,
     .		slen, nn, mb_allocated(pmax_mem_blks), i, iblk
	CHARACTER LEFINT*8, buff*40, b12*12
	PARAMETER (b12 = '            ')    !kob 12/96

************************************
* 11/16 brute force check of memory table list integrity
	DO mv = 1, max_mrs
	   mv_accounted_for(mv) = .FALSE.
	ENDDO
* ... free list (only forward mv links)
	n = 0
	mvfree = 0
	mv = mv_flink( 0 )	! base pointer for free slots
 10	IF ( mv .NE. 0 ) THEN
	   n = n + 1  ! number of passes
	   IF (n .GT. max_mrs + 5) THEN
	      CALL WARN('SANITY ERR: Trapped in free list ...')
	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL WARN('SANITY ERR: free mr list is corrupted: '
     .	   			//LEFINT(mv,slen)//tag )  ! loop forever
	   ENDIF
	   mv_accounted_for(mv) = .TRUE.
	   mvfree = mvfree + 1
	   mv = mv_flink(mv)
	   GOTO 10
	ENDIF
* ... hash piles (cached mvars)
	mvused = 0
	DO 14 pile = 1, pmv_npiles
	   last = -pile
	   mv = mv_flink( last )	! base pointer for slots in this hash #
	   n = 0
 12	   IF ( mv .GT. 0 ) THEN
	      n = n + 1  ! number of passes
	      IF (n .GT. max_mrs + 5) THEN
	         CALL WARN('SANITY ERR: Trapped in hash pile ...'
     .			//LEFINT(-pile,slen))
	      ENDIF
	      IF (mv_accounted_for(mv)) THEN
	         CALL WARN
     .			('SANITY ERR: cached mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//LEFINT(-pile,slen)//tag )
	      ENDIF
	      IF (mv_blink(mv) .NE. last) THEN
	         CALL WARN ('SANITY ERR: hash pile back link error: '
     .	   			//LEFINT(mv,slen)//LEFINT(-pile,slen)//tag )
	      ENDIF
	      mv_accounted_for(mv) = .TRUE.
	      mvused = mvused + 1
	      last = mv
	      mv = mv_flink(mv)
	      GOTO 12
	   ENDIF
 14	CONTINUE
* ... work space mvars
	n = 0
	mvwork = 0
	last = pmv_work_var_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 16	IF ( mv .GT. 0 ) THEN
	   n = n + 1  ! number of passes
	   IF (n .GT. max_mrs + 5) THEN
	      CALL WARN('SANITY ERR: Trapped in work space chain ...')
	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL WARN
     .		('SANITY ERR: work var mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	   IF (mv_blink(mv) .NE. last) THEN
	      CALL WARN('SANITY ERR: work var back link error: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	   mv_accounted_for(mv) = .TRUE.
	   mvwork = mvwork + 1
	   last = mv
	   mv = mv_flink(mv)
	   GOTO 16
	ENDIF
* ... uncached mvars
	n = 0
	mvuncached = 0
	last = pmv_nocache_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 18	IF ( mv .GT. 0 ) THEN
	   n = n + 1  ! number of passes
	   IF (n .GT. max_mrs + 5) THEN
	      CALL WARN('SANITY ERR: Trapped in uncached chain ...')
	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL WARN
     .		('SANITY ERR: uncached mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	   IF (mv_blink(mv) .NE. last) THEN
	      CALL WARN('SANITY ERR: uncached back link error: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	   mv_accounted_for(mv) = .TRUE.
	   mvuncached = mvuncached + 1
	   last = mv
	   mv = mv_flink(mv)
	   GOTO 18
	ENDIF
	IF (mvused+mvfree+mvwork+mvuncached .NE. max_mr_avail) THEN
	   CALL WARN
     .		('SANITY ERR: number of mr slots doesnt tally')
	ENDIF
	DO mv = 1, max_mr_avail
	   IF( .NOT.mv_accounted_for(mv)) THEN
	      CALL WARN('SANITY ERR: mr unaccounted in any list: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	ENDDO

* ... the deletion priority chain
	n = 0
	last = 0
	mv = mr_del_flink(last)
 19	IF ( mv .NE. 0 ) THEN
	   n = n + 1  ! number of passes
	   IF (n .GT. max_mrs + 5) THEN
	      CALL WARN('SANITY ERR: Trapped in deleted chain ...'//tag)
	      STOP 
	   ENDIF
	   IF (mr_del_blink(mv) .NE. last) THEN
	      CALL WARN ('SANITY ERR: deletion chain back link error: '
     .	   			//LEFINT(mv,slen)//tag )
	   ENDIF
	   last = mv
	   mv = mr_del_flink(mv)
	   GOTO 19
	ENDIF


************************************
* initialize memory block allocation counts
	DO mb = 1, pmax_mem_blks
	   mb_allocated(mb) = 0
	ENDDO

* check available bulk memory
	n      = 0	! number of contiguous regions
	mbfree = 0	! number of free blocks
	mbmax  = 0	! biggest contiguous region
	mb = mblk_flink( 0 )
 100	IF ( mb .NE. 0 ) THEN
	   n      = n + 1
	   size   = mblk_size( mb ) 
	   mbmax  = MAX( mbmax, size )
	   mbfree = mbfree + size
* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* ... no gaps should exist between sized regions - both free and used are sized
	   test_next = mb + size
	   IF ( test_next .EQ. max_mem_blks + 1 ) THEN
	      CONTINUE
	   ELSEIF ( test_next .GT. max_mem_blks + 1 
     .     .OR.     mblk_size(test_next) .EQ. unspecified_int4 ) THEN
	      CALL WARN('SANITY ERR:  improper region size: '
     .				//LEFINT(mb,slen)//tag )
	   ENDIF
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
* flag memory blocks allocated to the free list
	   DO i = 1, size
	      iblk = mb+i-1
!	      IF (iblk .EQ.878) WRITE (6,*) '*** block 878 is free'
	      mb_allocated(iblk) = mb_allocated(iblk) + 1
	   ENDDO

	   mb = mblk_flink( mb )
	   GOTO 100
	ENDIF	   

* check slots in memory variable table
* ... free table slots
	mvfree = 0
	mv     = mv_flink( 0 )	! base pointer for free slots
 200	IF ( mv .NE. 0 ) THEN

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
	   IF ( mr_protected(mv) .NE. mr_deleted ) THEN
	      CALL WARN('SANITY ERR: improperly deleted: '
     .				//LEFINT(mv,slen)//tag )
	   ENDIF
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	   mvfree = mvfree + 1
	   mv = mv_flink( mv )
	   GOTO 200
	ENDIF

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* check all the variables in all the hash piles
	mvused = 0
	mbused = 0
	DO 310 pile = 1, pmv_npiles
	   last = -pile
	   mv = mv_flink( last )	! base pointer for slots in this hash #
 300	   IF ( mv .GT. 0 ) THEN
	      IF ( mr_protected(mv) .EQ. mr_deleted
     .	      .OR. mr_nblks    (mv) .NE. mblk_size( mr_blk1(mv) )
     .	      .OR. mv_blink(mv) .NE. last ) THEN
	          CALL WARN('SANITY ERR: improperly cataloged: '
     .				//LEFINT(mv,slen)//tag )
	      ENDIF
	      mvused = mvused + 1
	      IF (mr_protected(mv) .NE. mr_table_entry_only) THEN
	         mbused = mbused + mr_nblks(mv)
	      ENDIF

* flag memory blocks allocated to a variable in a hash pile
	      IF (mr_protected(mv) .NE. mr_table_entry_only) THEN
	         DO i = 1, mr_nblks(mv)
	            iblk = mr_blk1(mv)+i-1
!	            IF (iblk .EQ.878) WRITE (6,*) '*** block 878 owned by',mv
	            mb_allocated(iblk) = mb_allocated(iblk) + 1
	         ENDDO
	      ENDIF

	      last   = mv
	      mv     = mv_flink(mv)
	      GOTO 300
	   ENDIF
 310	CONTINUE

* flag memory blocks allocated to work storage
	   DO i = 1, sp_num_blocks
	      iblk = sp_start_block+i-1
!	      IF (iblk .EQ.878) WRITE (6,*) '*** block 878 is work storage',
!     .		   sp_start_block, sp_num_blocks
	      mb_allocated(iblk) = mb_allocated(iblk) + 1
	   ENDDO


* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* check non-cached variables (10/99)
	nn = 0
	last = pmv_nocache_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 350	IF ( mv .GT. 0 ) THEN
	   IF ( mr_protected(mv) .EQ. mr_deleted
     .	   .OR. mr_protected(mv) .EQ. mr_in_progress
     .	   .OR. mr_nblks    (mv) .NE. mblk_size( mr_blk1(mv) )
     .	   .OR. mv_blink(mv) .NE. last ) THEN
	      CALL WARN('SANITY ERR: uncached variable error: '
     .				//LEFINT(mv,slen)//tag )
	   ENDIF
	   nn = nn + 1
	   mvused = mvused + 1
	   IF (mr_protected(mv) .NE. mr_table_entry_only) THEN
	      mbused = mbused + mr_nblks(mv)
	   ENDIF
!	   mbused = mbused + mr_nblks(mv)

* flag memory blocks allocated to a non-cached variable
	   IF (mr_protected(mv) .NE. mr_table_entry_only) THEN
	      DO i = 1, mr_nblks(mv)
	         iblk = mr_blk1(mv)+i-1
!	         IF (iblk .EQ.878) WRITE (6,*) '*** block 878 owned by',mv
	         mb_allocated(iblk) = mb_allocated(iblk) + 1
	      ENDDO
	   ENDIF

	   last   = mv
	   mv     = mv_flink(mv)
	   GOTO 350
	ENDIF

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* make sure no table slots or memory blocks have been lost
	nmvlost = max_mr_avail - ( mvfree+mvused )
	nmblost = max_mem_blks   - ( mbfree+mbused+sp_num_blocks )
	IF ( nmvlost .NE. 0 ) THEN
	   CALL WARN('SANITY ERR: lost table slots: '
     .				//LEFINT(nmvlost,slen)//tag )
	ENDIF
	IF ( nmblost .NE. 0 ) THEN
	   CALL WARN('SANITY ERR: lost memory blocks: '
     .				//LEFINT(nmblost,slen)//tag )
	ENDIF

* * * * integrity check * * * * * * * * * * * * * * * * * * * * *
* make sure that every memory block is allocated to exactly one purpose
	DO mb = 1, pmax_mem_blks
	   IF (mb_allocated(mb) .NE. 1) THEN
	      CALL WARN('SANITY ERR: allocation of block: '
     .				//LEFINT(mb,slen)//
     .			  ' is '//LEFINT(mb_allocated(mb),slen)//tag )
	   ENDIF
	ENDDO

* * * * integrity check * * * * * * * * * * * * * * * * * * * * *
* variable deletion priority chain
	navail = 0
	mv = mr_del_flink(0)
 400	IF ( mv .NE. 0 ) THEN
	   navail = navail + 1
	   IF ( mr_protected(mv) .NE. mr_not_protected ) THEN
	      CALL WARN('SANITY ERR: Protected variables deleted')
	   ENDIF
	   mv = mr_del_flink(mv)
	   GOTO 400
	ENDIF
	ninuse = 0
	ndel   = 0
	nperm  = 0
	mbperm = 0
	ntemp  = 0
        ninprogress = 0
	ntable_entries = 0
	DO 410 mv = 1, max_mr_avail
	   IF ( mr_protected(mv) .GT. 0 ) THEN
	      ninuse = ninuse + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_perm_protected ) THEN
	      nperm  = nperm  + 1
	      mbperm = mbperm + mr_nblks(mv)
	   ELSEIF ( mr_protected(mv) .EQ. mr_deleted ) THEN
	      ndel = ndel + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_temporary ) THEN
	      ntemp = ntemp + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_in_progress ) THEN
	      ninprogress = ninprogress + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_table_entry_only ) THEN
	      ntable_entries = ntable_entries + 1
	   ENDIF
 410	CONTINUE
!	IF ( nperm .GT. 0 ) THEN
!	   CALL SPLIT_LIST( pttmode_explct, lun,
!     .			 b12//'number of /PERMANENT variables: '//
!     .			 LEFINT(nperm,slen), 0 )
!	   CALL SPLIT_LIST( pttmode_explct, lun,
!     .			 b12//'number of blocks used /PERM: '//
!     .			 LEFINT(mbperm,slen), 0 )
!	ENDIF
	nmvlost = max_mr_avail
     .		- (ninuse+navail+nperm+ndel+ntemp+ninprogress+ntable_entries)
	IF ( nmvlost .NE. 0 ) THEN
	   CALL WARN( 'SANITY ERR: Unaccountable var protection: '
     .				//LEFINT(nmvlost,slen)//tag )
	ENDIF
	n = ndel - mvfree
	IF ( n .NE. 0 ) THEN
	   CALL WARN( 'SANITY ERR: Deleted vars unaccountable: '
     .				//LEFINT(nmvlost,slen)//tag )
	ENDIF

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	RETURN
	END


