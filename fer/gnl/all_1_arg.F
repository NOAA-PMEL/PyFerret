	SUBROUTINE ALL_1_ARG

* treat all the arguments identified by PARSE_COMMAND as a single argument
* example cases that this has to address:
*	message "hello world" 
*	message "hello" "world"
*	message " "hello" "world" "     - unclear what to do ...

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V300: 4/22/93

* calling argument declarations:

* internal variable declarations:
	INTEGER i

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* reset the command common to look like a single big argument

	IF ( num_args .GT. 1 ) THEN

	   arg_end(1) = arg_end(num_args)

* protect against lost " from start of:  message "hello" "world"
	   i = arg_start(1) - 1
	   IF ( cmnd_buff(i:i) .EQ. '"' ) arg_start(1) = i

* protect against lost " from end of:  message "hello" "world"
	   i = arg_end(1) + 1
	   IF ( i .LE. len_cmnd ) THEN
	      IF ( cmnd_buff(i:i) .EQ. '"' ) arg_end(1) = i
	   ENDIF

	   num_args = 1

	ENDIF

	RETURN
	END
