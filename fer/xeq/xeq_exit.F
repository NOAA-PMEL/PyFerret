	SUBROUTINE XEQ_EXIT( * )

* execute the EXIT command
* if program is not under interactive control the /COMMAND_FILE qualifier is
* required

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/3/86
* revision 1.0 - 4/30/87 - EXIT in middle of command file will work
* revision 1.1 - 4/23/87 - flag arguments given as errors
* V312: 5/94 - use abnormal return for final program exit
* V314: 8/24/94 *kob* IBM port: slash_command_file should be declared as 
*				logical
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

	INTEGER status
	LOGICAL slash_command_file

* make sure no arguments were given
	IF ( num_args .GT. 0 ) CALL ERRMSG( ferr_invalid_subcmnd, 
     .			status, cmnd_buff(arg_start(1):len_cmnd), *5000 )

* real exit or just from command file ?
	slash_command_file =  num_qualifiers    .EQ. 1
     .			.AND. qualifier_list(1) .EQ. 1

	IF ( input_source .NE. input_from_file
     .	.OR. slash_command_file			) THEN

* exit the program
	   CALL CLSPPL						! close PLOT5
!	   IF ( frame_on ) CALL MOVIE_CLOSE( frame_lun, status )! end movie
	   RETURN 1

	ELSE

* exit the command file, only
	   CALL POP_CS
	   RETURN

	ENDIF
	   
* error exit
 5000	RETURN

	END

