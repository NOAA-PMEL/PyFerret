	SUBROUTINE XEQ_HELP

* execute the HELP command

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86 - dummy version
* revision 1.0 - 5/13/87 - calls VMS utility routines
* revision 1.1 - 3/7/88  - changed help file name to logical name
* revision 1.2 - 4/26/88 - added PUT_OUTPUT/GET_INPUT
* revision 1.3 - 6/22/88 - changed GFDL_HELP --> FERRET_HELP
* revision 1.4 - 8/3/88  - Top level of help library, "FERRET", removed
* V230: 8/26/92 - added Unix version - entirely separate from VMS
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI


#ifdef unix
        include 'ferret_cmn/ferret.parm'
        include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'FERRET_CMN:XERRMSG_TEXT.CMN'
	INCLUDE	'FERRET_CMN:XLBR_IO.CMN'
	INCLUDE '($HLPDEF)'

	EXTERNAL PUT_OUTPUT, GET_INPUT
#endif

#ifdef unix
	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .   ' Detailed help is available via the Unix command Fhelp.'
     .								, 0)
	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .   ' To execute Fhelp from the "yes?" prompt type   SPAWN Fhelp'
     .								, 0)
	CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .   ' To obtain a list of FERRET commands type    SHOW COMMANDS'
     .								, 0)

        RETURN
#else

* internal variable declarations:
	INTEGER		LBR$OUTPUT_HELP, PUT_OUTPUT, GET_INPUT
	INTEGER		status, vax_code
	CHARACTER*24	ferret_help_lib / 'FERRET_HELP' /	! logical name
	CHARACTER*16	err_code

* initialize page control
	line_count = 0
	stop_output = .FALSE.

* call for HELP in the method requested
	IF ( num_args .EQ. 0 ) THEN

* just "HELP" was cried out
	WRITE ( ttout_lun, * )
     .	'Use "HELP error" to get help on the most recent ',
     .						'program error message'
	   vax_code = LBR$OUTPUT_HELP(
     .				PUT_OUTPUT,
     .				,
     .				' ',
     .				ferret_help_lib,
     .				HLP$M_PROMPT,
     .				GET_INPUT	)

	ELSEIF ( num_args .EQ. 1 .AND.
     .		 cmnd_buff( arg_start(1):arg_start(1)+2 ) .EQ. 'err' .OR.
     .		 cmnd_buff( arg_start(1):arg_start(1)+2 ) .EQ. 'ERR' ) THEN

* "HELP error"
	   IF     ( last_error .GT. greport   ) THEN
	      err_code = report_err_code( last_error - greport   )
	   ELSEIF ( last_error .GT. gnoreport ) THEN
	      err_code = noreport_err_code( last_error - gnoreport )
	   ELSEIF ( last_error .GT. gspecial ) THEN
	      err_code = special_err_code( last_error - gspecial  )
	   ELSE
              err_code = 'unknown_err'
	   ENDIF
	   vax_code = LBR$OUTPUT_HELP(
     .				PUT_OUTPUT,
     .				,
     .				'error '//err_code,
     .				ferret_help_lib,
     .				0,
     .				GET_INPUT	)
	
	ELSE
* "HELP somethin-or-other" was given
	   vax_code = LBR$OUTPUT_HELP(
     .				PUT_OUTPUT,
     .				,
     .				cmnd_buff(arg_start(1):len_cmnd),
     .				ferret_help_lib,
     .				HLP$M_PROMPT,
     .				GET_INPUT	)
	ENDIF

	IF ( MOD(vax_code,2) .EQ. 0 ) CALL ERRMSG
     .			( ferr_internal, status, ferret_help_lib, *5000 )
	RETURN

* error exit
 5000	RETURN
#endif
	END

