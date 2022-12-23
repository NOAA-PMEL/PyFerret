	SUBROUTINE CHECK_MEMORY ( lun )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/20/88
* revision 0.1 - 8/15/88 - max_mrs --> max_mr_avail
* V200:  6/9/89 - deleted variables flagged in mr_protected
*		- deletable variables in mr_del_*
*		- slight reformatting of output
*	 5/3/90 - include temporary vars in del_link chain check
* V313:8/18/94 *kob* IBM Port - format statement 3310 was missing a comma
* V320:8/11/94 - allow bigger output field for (resizable) mem_blk_size
*     12/29/94 - use SPLIT_LIST to redirect tty output for GUI
*		==>  extensive changes!!
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line.
* V510: 10/99 *sh* - make checks on 
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
* V702 *sh* Added dynamic memory checks.  Removed old block-oriented stuff
C 12/2022 *acm* Code cleanup: Call LEFINT8 for INTEGER*8 arguments

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'xvariables.cmn'

* calling argument declarations:
	INTEGER	lun

* internal variable declarations:
	INTEGER	mv, mvfree, mvused, 
     .		nmvlost, pile, last, size, test_next,
     .		navail, nperm, ntemp, ndel, slen, nn, mrsizelen
        INTEGER*8 mem_in_use
	CHARACTER LEFINT*8, LEFINT8*8, buff*40, b04*4, mrsizebuf*16, byte_unit*2
	PARAMETER (b04 = '    ')


* check Ferret variable memory allocations
        mem_in_use = 0
        DO mv = 1, max_mrs
           mem_in_use = mem_in_use + mr_size(mv)
        ENDDO
* ... no valid ws_size() work memory in use when this routine is called
        IF (mem_in_use .NE. total_mem) THEN
	      CALL WARN('***** corrupted total mem size: '
     .                  //LEFINT8(mem_in_use,slen) )
        ENDIF
	CALL SPLIT_LIST( pttmode_explct, lun, 
     .			 b04//'Total table slots: '//
     .			 LEFINT(max_mr_avail,slen), 0 )


* check slots in memory variable table
* ... free table slots
	mvfree = 0
	mv     = mv_flink( 0 )	! base pointer for free slots
 200	IF ( mv .NE. 0 ) THEN

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
	   IF ( mr_protected(mv) .NE. mr_deleted ) CALL WARN
     .			('CORRUPTION! improperly deleted: '
     .						//LEFINT(mv,slen) )
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

	   mvfree = mvfree + 1
	   mv = mv_flink( mv )
	   GOTO 200
	ENDIF
	CALL SPLIT_LIST( pttmode_explct, lun,
     .			 b04//'Free table slots: '//
     .			 LEFINT(mvfree,slen), 0 )

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* check all the variables in all the hash piles
	mvused = 0
	DO 310 pile = 1, pmv_npiles
	   last = -pile
	   mv = mv_flink( last )	! base pointer for slots in this hash #
 300	   IF ( mv .GT. 0 ) THEN
	      IF ( mr_protected(mv) .EQ. mr_deleted
     .	      .OR. mr_protected(mv) .EQ. mr_in_progress
     .	      .OR. mv_blink(mv) .NE. last ) 	CALL WARN
     .			('CORRUPTION! improperly cataloged: '
     .						//LEFINT(mv,slen) )
	      mvused = mvused + 1
	      last   = mv
	      mv     = mv_flink(mv)
	      GOTO 300
	   ENDIF
 310	CONTINUE


* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* check non-cached variables (10/99)
	nn = 0
	last = pmv_nocache_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 350	IF ( mv .GT. 0 ) THEN
	   IF ( mr_protected(mv) .EQ. mr_deleted
     .	   .OR. mr_protected(mv) .EQ. mr_in_progress
     .	   .OR. mv_blink(mv) .NE. last ) 	CALL WARN
     .			('CORRUPTION! uncached variable error: '
     .						//LEFINT(mv,slen) )
	   nn = nn + 1
	   mvused = mvused + 1
	   last   = mv
	   mv     = mv_flink(mv)
	   GOTO 350
	ENDIF
	CALL SPLIT_LIST( pttmode_explct, lun,
     .			 b04//'Un-cached variables: '//
     .			 LEFINT(nn,slen), 0 )

* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* check working storage (10/99)
	nn = 0
	last = pmv_work_var_pile
	mv = mv_flink( last )	! base pointer for slots in this hash #
 360	IF ( mv .GT. 0 ) THEN
	   nn = nn + 1
	   mvused = mvused + 1
	   last   = mv
	   mv     = mv_flink(mv)
	   GOTO 360
	ENDIF
	IF ( nn .GT. 0 ) CALL WARN
     .			('CORRUPTION! unfreed work arrays: '
     .						//LEFINT(nn,slen) )


* * * * integrity check * * * * * * * * * * * * * * * * * * * *
* make sure no table slots or memory blocks have been lost
	nmvlost = max_mr_avail - ( mvfree+mvused )
	IF ( nmvlost .NE. 0 ) CALL WARN
     .			('CORRUPTION! lost table slots: '
     .						//LEFINT(nmvlost,slen) )

* * * * integrity check * * * * * * * * * * * * * * * * * * * * *
* variable deletion priority chain
	navail = 0
	mv = mr_del_flink(0)
 400	IF ( mv .NE. 0 ) THEN
	   navail = navail + 1
	   IF ( mr_protected(mv) .NE. mr_not_protected ) CALL WARN
     .			('CORRUPTION! Protected variables deleted')
	   mv = mr_del_flink(mv)
	   GOTO 400
	ENDIF
	ndel   = 0
	nperm  = 0
	ntemp  = 0
        mem_in_use = 0
	DO 410 mv = 1, max_mr_avail
	   IF ( mr_protected(mv) .EQ. mr_perm_protected ) THEN
	      nperm  = nperm  + 1
              mem_in_use = mem_in_use + mr_size(mv)
	   ELSEIF ( mr_protected(mv) .EQ. mr_deleted ) THEN
	      ndel = ndel + 1
	   ELSEIF ( mr_protected(mv) .EQ. mr_temporary ) THEN
	      ntemp = ntemp + 1
	   ENDIF
 410	CONTINUE
	IF ( nperm .GT. 0 ) THEN
	   CALL SPLIT_LIST( pttmode_explct, lun,
     .			 b04//'number of /PERMANENT variables: '//
     .			 LEFINT(nperm,slen), 0 )
	   	   
	   CALL MEM_TO_BYTES (mem_in_use, mrsizebuf, slen, byte_unit)

	   CALL SPLIT_LIST( pttmode_explct, lun,
     .			 b04//'memory used for /PERM: '//
     .			 mrsizebuf(:slen)//byte_unit, 0 )
	ENDIF
	nmvlost = max_mr_avail - (navail+nperm+ndel+ntemp)
	IF ( nmvlost .NE. 0 ) CALL WARN
     .		( 'CORRUPTION! Protected vars unaccountable: '
     .				//LEFINT(nmvlost,slen) )
	nn = ndel - mvfree
	IF ( nn .NE. 0 ) CALL WARN
     .		( 'CORRUPTION! Deleted vars unaccountable: '
     .				//LEFINT(nmvlost,slen) )

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	RETURN
	END


