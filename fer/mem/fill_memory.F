	SUBROUTINE FILL_MEMORY ( memory, start_block, num_blocks, fill_word )

* fill a memory region with the specified value

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/23/86

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	start_block, num_blocks
	REAL	memory( mem_blk_size, max_mem_blks )
	REAL	fill_word

* internal variable declarations:
	INTEGER	iword, iblock

	DO 100 iblock = start_block, start_block+num_blocks-1
	DO 100 iword  = 1, mem_blk_size
 100	memory( iword, iblock ) = fill_word	

	RETURN
	END

