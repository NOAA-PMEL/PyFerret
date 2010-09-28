	SUBROUTINE INIT_MEMORY( block_size, number_of_blocks )

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

	include 'tmap_dims.parm'
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include 'xvariables.cmn'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER block_size, number_of_blocks

* internal variable declarations
	INTEGER	bigblk, mv
      INTEGER i1, i2, len
      CHARACTER*30 TM_FMT, sym_name, buff

* initialize memory configuration
	mem_blk_size = block_size
	max_mem_blks = number_of_blocks

* BULK MEMORY
* note: arrays mblk_* are initialized by DATA to unspecified
* ... designate all the rest as 1 giant free contiguous block
	bigblk = 1
	mblk_flink(0) = bigblk
	mblk_blink(0) = bigblk
	mblk_size (0) = 0
	mblk_flink(bigblk) = 0
	mblk_blink(bigblk) = 0
	mblk_size (bigblk) = max_mem_blks

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
	mode_arg(pmode_desperate,1) = 0.1 *  block_size *  number_of_blocks

	RETURN
	END

