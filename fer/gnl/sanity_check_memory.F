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
* note that this routine could be streamlined -- the linked lists are
* traversed two separate times.  Since this code is only run in diagnostic
* situations we have not troubled to mske it nicer.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* V702: *sh* 11/16 This is an expanded version of CHECK_MEMORY intended 
*      for use with -D debug copilations
*            1/17 - output to stdout for easier debugging
C 12/2022 *acm* Code cleanup: Call LEFINT8 for INTEGER*8 arguments

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'
	include 'xwork_space.cmn'

* calling argument declarations:
	INTEGER	lun
	CHARACTER*(*) tag

* internal variable declarations:
	LOGICAL mv_accounted_for(max_mrs)
	INTEGER	mv, mvfree, mvused, mvuncached, mvwork,
     .		nmvlost, n, pile, last, msize, test_next,
     .		ninuse, navail, nperm, mbperm, ntemp, ndel,
     .		ninprogress, ntable_entries, 
     .		slen, nn, i, ws, s1, s2
	INTEGER*8 sum, mem_in_use
	CHARACTER LEFINT*10, LEFINT8*10, buff*40, b12*12
	PARAMETER (b12 = '            ')    !kob 12/96

************************************
* check Ferret variable memory allocations
        mem_in_use = 0
        DO mv = 1, max_mrs
           mem_in_use = mem_in_use + mr_size(mv)
        ENDDO
        IF (mem_in_use .NE. total_mem) THEN
	      CALL WARN('***** corrupted total mem size: '
     .                  //LEFINT8(mem_in_use,slen) )
        ENDIF

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
	      CALL TM_NOTE('SANITY ERR: Trapped in free list ...'
     .				, ttout_lun)
	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL TM_NOTE('SANITY ERR: free mr list is corrupted: '
     .	   		//LEFINT(mv,slen)//tag, ttout_lun)   ! loop forever
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
	         CALL TM_NOTE('SANITY ERR: Trapped in hash pile ...'
     .			//LEFINT(-pile,slen), ttout_lun) 
	      ENDIF
	      IF (mv_accounted_for(mv)) THEN
	         CALL TM_NOTE
     .			('SANITY ERR: cached mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//LEFINT(-pile,slen)//tag,
     .				ttout_lun) 
	      ENDIF
	      IF (mv_blink(mv) .NE. last) THEN
	         CALL TM_NOTE ('SANITY ERR: hash pile back link error: '
     .	   			//LEFINT(mv,slen)//LEFINT(-pile,slen)//tag,
     .				ttout_lun) 
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
	      CALL TM_NOTE('SANITY ERR: Trapped in work space chain ...',
     .				ttout_lun) 

	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL TM_NOTE
     .		('SANITY ERR: work var mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun) 
	   ENDIF
	   IF (mv_blink(mv) .NE. last) THEN
	      CALL TM_NOTE('SANITY ERR: work var back link error: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun)
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
	      CALL TM_NOTE('SANITY ERR: Trapped in uncached chain ...',
     .				ttout_lun)	
	   ENDIF
	   IF (mv_accounted_for(mv)) THEN
	      CALL TM_NOTE
     .		('SANITY ERR: uncached mr slot found in multiple lists: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	   IF (mv_blink(mv) .NE. last) THEN
	      CALL TM_NOTE('SANITY ERR: uncached back link error: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	   mv_accounted_for(mv) = .TRUE.
	   mvuncached = mvuncached + 1
	   last = mv
	   mv = mv_flink(mv)
	   GOTO 18
	ENDIF
	IF (mvused+mvfree+mvwork+mvuncached .NE. max_mr_avail) THEN
	   CALL TM_NOTE
     .		('SANITY ERR: number of mr slots doesnt tally', ttout_lun)
	ENDIF
	DO mv = 1, max_mr_avail
	   IF( .NOT.mv_accounted_for(mv)) THEN
	      CALL TM_NOTE('SANITY ERR: mr unaccounted in any list: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	ENDDO

* ... the deletion priority chain
	n = 0
	last = 0
	mv = mr_del_flink(last)
 19	IF ( mv .NE. 0 ) THEN
	   n = n + 1  ! number of passes
	   IF (n .GT. max_mrs + 5) THEN
	      CALL TM_NOTE('SANITY ERR: Trapped in deleted chain ...'
     .				//tag, ttout_lun)
	      STOP 
	   ENDIF
	   IF (mr_del_blink(mv) .NE. last) THEN
	      CALL TM_NOTE ('SANITY ERR: deletion chain back link error: '
     .	   			//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	   last = mv
	   mv = mr_del_flink(mv)
	   GOTO 19
	ENDIF


************************************

* check slots in memory variable table
* ... free table slots
	mvfree = 0
	mv     = mv_flink( 0 )	! base pointer for free slots
 200	IF ( mv .NE. 0 ) THEN

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
	   IF ( mr_protected(mv) .NE. mr_deleted ) THEN
	      CALL TM_NOTE('SANITY ERR: improperly deleted: '
     .				//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	   mvfree = mvfree + 1
	   mv = mv_flink( mv )
	   GOTO 200
	ENDIF

* * * * SUMMARIZE MEMORY TABLE SLOTS IN USE * * * *
* check all the variables in all the hash piles
	mvused = 0
	DO 310 pile = 1, pmv_npiles
	   last = -pile
	   mv = mv_flink( last )	! base pointer for slots in this hash #
 300	   IF ( mv .GT. 0 ) THEN
	      IF ( mr_protected(mv) .EQ. mr_deleted
     .	      .OR. mv_blink(mv) .NE. last ) THEN
	          CALL TM_NOTE('SANITY ERR: improperly cataloged: '
     .				//LEFINT(mv,slen)//tag, ttout_lun)
	      ENDIF
	      mvused = mvused + 1
	      last   = mv
	      mv     = mv_flink(mv)
	      GOTO 300
	   ENDIF
 310	CONTINUE

* check non-cached variables (10/99)
	last = pmv_nocache_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 350	IF ( mv .GT. 0 ) THEN
	   IF ( mr_protected(mv) .EQ. mr_deleted
     .	   .OR. mr_protected(mv) .EQ. mr_in_progress
     .	   .OR. mv_blink(mv) .NE. last ) THEN
	      CALL TM_NOTE('SANITY ERR: uncached variable error: '
     .				//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	   mvused = mvused + 1
	   last   = mv
	   mv     = mv_flink(mv)
	   GOTO 350
	ENDIF

* check work storage allocated by GET_WORK_MR
	last = pmv_work_var_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 370	IF ( mv .GT. 0 ) THEN
	   IF ( mr_protected(mv) .NE. mr_in_progress
     .	   .OR. mv_blink(mv) .NE. last ) THEN
	      CALL TM_NOTE('SANITY ERR: table work storage variable error: '
     .				//LEFINT(mv,slen)//tag, ttout_lun)
	   ENDIF
	   mvused = mvused + 1
	   last   = mv
	   mv     = mv_flink(mv)
	   GOTO 370
	ENDIF

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* make sure no table slots or memory blocks have been lost
	nmvlost = max_mr_avail - ( mvfree+mvused )
	IF ( nmvlost .NE. 0 ) THEN
	   CALL TM_NOTE('SANITY ERR: lost table slots: '
     .				//LEFINT(nmvlost,slen)//tag, ttout_lun)
	ENDIF

* * * * integrity check * * * * * * * * * * * * * * * * * * * * *
* variable deletion priority chain
	navail = 0
	mv = mr_del_flink(0)
 400	IF ( mv .NE. 0 ) THEN
	   navail = navail + 1
	   IF ( mr_protected(mv) .NE. mr_not_protected ) THEN
	      CALL TM_NOTE('SANITY ERR: Protected variables deleted',
     .				ttout_lun)
	   ENDIF
	   mv = mr_del_flink(mv)
	   GOTO 400
	ENDIF
	ninuse = 0
	ndel   = 0
	nperm  = 0
	ntemp  = 0
        ninprogress = 0
	ntable_entries = 0
	DO 410 mv = 1, max_mr_avail
	   IF ( mr_protected(mv) .GT. 0 ) THEN
	      ninuse = ninuse + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_perm_protected ) THEN
	      nperm  = nperm  + 1
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
	nmvlost = max_mr_avail
     .		- (ninuse+navail+nperm+ndel+ntemp+ninprogress+ntable_entries)
	IF ( nmvlost .NE. 0 ) THEN
	   CALL TM_NOTE( 'SANITY ERR: Unaccountable var protection: '
     .				//LEFINT(nmvlost,slen)//tag, ttout_lun)
	ENDIF
	n = ndel - mvfree
	IF ( n .NE. 0 ) THEN
	   CALL TM_NOTE( 'SANITY ERR: Deleted vars unaccountable: '
     .				//LEFINT(nmvlost,slen)//tag, ttout_lun)
	ENDIF

* * * * * * dynamic memory sanity check * * * * *
	sum = 0
	DO mv = 1, max_mr_avail
	   s1 = mr_size(mv)
	   s2 = SIZE(memry(mv)%ptr) 
	   sum = sum + s1
	   IF ( mr_protected(mv) .NE. mr_table_entry_only) THEN
              IF (s1 .NE. s2) THEN
	         CALL TM_NOTE( 'SANITY ERR: Dynamic mem size error: '
     .			//LEFINT(mv,slen)//LEFINT(s1,slen)
     .			//LEFINT(s2,slen)//tag, ttout_lun)
	      ENDIF
	   ELSE
              IF (s1 .NE. 0) THEN
	         CALL TM_NOTE( 'SANITY ERR: Table entry size error: '
     .			//LEFINT(mv,slen)//LEFINT(s1,slen)
     .			//LEFINT(s2,slen)//tag, ttout_lun)
	      ENDIF
	   ENDIF
	ENDDO
	IF (sum .NE. total_mem) THEN
	   CALL TM_NOTE( 'SANITY ERR: Dynamic mem total: '
     .			//LEFINT8(sum,slen)//LEFINT8(total_mem,slen)//tag,
     .			 ttout_lun)
	ENDIF

* work storage
	DO ws = 1, max_ws
	   s1 = ws_size(ws)
	   s2 = SIZE(workmem(ws)%ptr) 
           IF (s1 .NE. s2) THEN
	      CALL TM_NOTE( 'SANITY ERR: Dynamic work space size error: '
     .			//LEFINT(ws,slen)//LEFINT(s1,slen)
     .			//LEFINT(s2,slen)//tag, ttout_lun)
	   ENDIF
	ENDDO

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	RETURN
	END
