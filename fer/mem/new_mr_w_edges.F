	SUBROUTINE NEW_MR_W_EDGES(memory,cx_valid,cx_model,mr,status)

* allocate and claim sufficient memory space, and assign and fill a slot in the 
* memory variable table for the variable or expression indicated in the context
* the region of valid data within the cx_model context is indicated by the
* cx_valid context ... fill the invalid regions with bad data flags

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   8/3/89
* V312: 5/94 - array "memory" as a calling argument

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER	cx_valid, cx_model, mr, status
	REAL	memory( mem_blk_size, max_mem_blks )

* create the memory variable
	CALL CREATE_MEM_VAR( cx_model, mr, status )
	IF ( status .NE. ferr_ok ) RETURN

* fill edges
	CALL BAD_EDGES( cx_valid, memory(1, mr_blk1(mr)), mr )

	RETURN
	END
