	SUBROUTINE CREATE_MEM_VAR( cx, mr, status )

* allocate and claim sufficient memory space, and assign and fill a slot in the 
* memory variable table for the variable or expression indicated in the context

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		cx, mr, status

* internal variable declarations:
	INTEGER	CGRID_SIZE, grid_size, start_block, num_blocks

* amount of storage space required
	grid_size = CGRID_SIZE( cx )

* claim the storage space
	CALL GET_MEMORY( grid_size, start_block, num_blocks, status )
	IF ( status .NE. ferr_ok ) RETURN

* catalog it in the memory variable table
	CALL ASSIGN_VARIABLE
     .		      ( cx,
     .			start_block,
     .			num_blocks,
     .			mr,
     .			status 	)
	RETURN
	END
