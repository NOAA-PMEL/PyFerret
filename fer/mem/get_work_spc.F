	SUBROUTINE GET_WORK_SPC ( size, start_block, status )

* provide a region of temporary working space to the calling routine.
* For greatest speed simply reuse the last provided if it is large enough

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87

* calling argument declarations:
	INTEGER	size, start_block, status

* --- end of introductory code ---
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xvariables.cmn'
	include 'ferret_cmn/xwork_space.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XWORK_SPACE.CMN'
#endif

	IF ( size .LE. sp_last_size ) THEN
	   status = ferr_ok

	ELSE
	   CALL RELEASE_WORK_SPC
	   CALL GET_MEMORY( size, sp_start_block, sp_num_blocks, status )
	   sp_last_size = sp_num_blocks * mem_blk_size
	ENDIF

	start_block = sp_start_block
	RETURN
	END

