	SUBROUTINE GET_VMS_MODE ( mode )

* determine the operational mode under VMS: BAATCH,NETWORK,INTERACTIVE or OTHER

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/2/87

#ifdef unix
	INTEGER	mode
        STOP 'GET_VMS_MODE called'
#else
	INCLUDE '($JPIDEF)'

	INTEGER	mode

	STRUCTURE / GETJPI_ITEM /
		INTEGER*2 mode_code_len, mode_code
		INTEGER*4 mode_addr,mode_len/0/
	END STRUCTURE

	RECORD / GETJPI_ITEM / get_mode

	get_mode.mode_code_len  = 4
	get_mode.mode_code      = JPI$_MODE
	get_mode.mode_addr      = %LOC( mode )

	CALL SYS_GETJPIW( ,,,get_mode,,, )

	RETURN
#endif

	END
