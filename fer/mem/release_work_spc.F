	SUBROUTINE RELEASE_WORK_SPC

* releases claim to temporary work space allocated by GET_WORK_SPC

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/19/87

#ifdef unix
	include 'ferret_cmn/xwork_space.cmn'
#else
	INCLUDE 'FERRET_CMN:XWORK_SPACE.CMN'
#endif

	IF ( sp_last_size .GT. 0 ) THEN
	   CALL FREE_MEMORY( sp_start_block, sp_num_blocks )
	   sp_last_size = 0
	ENDIF

	RETURN
	END
