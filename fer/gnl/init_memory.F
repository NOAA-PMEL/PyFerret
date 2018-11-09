	SUBROUTINE INIT_MEMORY (vmem_size_arg)

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
* initialize memory management linked lists, etc.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/21/88
* revision 0.1 - 8/15/88 - only initialize to max_mrs-1 - top is dummy
* revision 0.2 -11/22/88 - new TM IO library eliminating nblks_for_lines
* V200:  6/12/89 - initialize back link of memory regions
* V312:  5/19/94 - changes for reconfigurable (dynamic) memory
* V420: initialize the no-cache memory hash pointer
* V500: 4/99 *sh* - set mode desperate while we're at it
* V510: 2/00 *sh* initialize additional new hash piles
* V606: 8/07 *acm* Set a symbol with the current memory setting
* V62   5/09 *acm*- make FERRET_MEMORY a special symbol (remove 
*                 code defining it from this routine)
* V702: 1/17 *sh* for trac enhancement #2369 -- dynamic memory management 
*                - perform initial sanity check
* V72 : 5/17 *acm* rmem_size comes in as-is; handle it as in xeq_set for a 
*                  SET MEM/SIZ=

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	REAL vmem_size_arg

* internal variable declarations
	INTEGER	bigblk, mv, ws
        INTEGER i1, i2, len
	INTEGER*8 i8_val
        CHARACTER*30 TM_FMT, sym_name, buff

* LINKS IN MEMORY VARIABLE TABLE
* ... all hash pile start pointers point to self (empty lists)
	DO 100 mv = pmv_lowest_head, -1
	   mv_flink(mv) = mv
	   mv_blink(mv) = mv
 100	CONTINUE
* ... free list (pointer at subscript 0) contains all table slots
	DO 110 mv = 0, max_mr_avail
	   mv_flink(mv) = mv + 1
	   mv_blink(mv) = unspecified_int4   ! bck link not used in free list
 110	CONTINUE
	mv_flink(max_mr_avail) = 0	     ! make list circular

* set MODE DESPERATE to 1/10 of full memory size
* the reason so small is that Ferret memory management is not aware of
* large hierarchies of definitions -- so a command like "LIST var"
* may involve far more data than the transformations of "var" alone
	mode_arg(pmode_desperate,1) = 0.1 *  pmax_mem_default

* initialize malloced dynamic memory management variables  (2/2017)
        DO mv = 1, max_mrs
           CALL NULLIFY_MR(mv)
           mr_size(mv) = 0
        ENDDO
        DO ws = 1, max_ws
           CALL NULLIFY_WS(ws)
           ws_size(ws) = 0
        ENDDO

* SET MEMORY limit default
* -memsize command-line argument passed in, handle as in xeq_set

	IF (vmem_size_arg .NE. 0.) THEN 
	   i8_val = (vmem_size_arg*1.0000000001D0) * 1000000  ! int tuncation issues ...
	   max_mem_allowed = i8_val 
	ELSE
	   max_mem_allowed = pmax_mem_default
	ENDIF

* accounting variables (see xvariables.cmn)
        total_mem = 0
        peak_mem = 0
        peak_essential_mem = 0
	need_memlim_warning = .TRUE.
	permanent_mem = 0
	essential_mem = 0

* sanity check
#ifndef NDEBUG
        CALL SANITY_CHECK_MEMORY("init_memory")
#endif

	RETURN
	END

