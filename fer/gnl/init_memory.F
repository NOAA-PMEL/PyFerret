	SUBROUTINE INIT_MEMORY( block_size, number_of_blocks )

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

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER block_size, number_of_blocks

* internal variable declarations
	INTEGER	bigblk, mv

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
	DO 100 mv = pmv_nocache_pile, -1
	   mv_flink(mv) = mv
	   mv_blink(mv) = mv
 100	CONTINUE
* ... free list (pointer at subscript 0) contains all table slots
	DO 110 mv = 0, max_mr_avail
	   mv_flink(mv) = mv + 1
	   mv_blink(mv) = unspecified_int4	! bck link not used in free list
 110	CONTINUE
	mv_flink(max_mr_avail) = 0		! make list circular

	RETURN
	END
