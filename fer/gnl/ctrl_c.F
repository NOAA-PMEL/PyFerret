	SUBROUTINE SET_CTRL_C (CTRLC_AST)

* turn on facility to trap keyboard interrupts
* the interrupt handler defined here simply sets the common variable
* "interrupted" to .TRUE.

* on VMS:
* queue a VMS "asynchronous system trap" to catch user interrupts via ^C
* note that this routine must be called again following each interrupt
* this code is based on page 8-47 of Guide to Programming on VAX/VMS (Fortran)

* on Unix:
* use routine "SIGNAL" to define the interrupt handler

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.00 - 11/27/87 SH based on QINTERRUPT from program PRNT 8/2/85
* Unix/RISC port 3/18/91

* V313 - 11/1/94 - AIX port: use "SIGNAL_" instead of "SIGNAL" and
*		must reset signal handler after every use

* v4.20	- 1/30/96 - modified ifdef logic for determining call to
*		    signal handler.  Turns out solaris also needs
*		    to call the signal handler each time.  *kob*
* 12/96 *kob* - Linux Port - had to move variable assignments off of
*               declaration line. Used DATA rather than PARAMETER
*		statement because variable modified later in routine
*	      - put a ! in front of "D" for disabled lines

* argument: CTRLC_AST
*	name of routine to be called if operator hits ^C
*	(or call with argument %VAL(0) to return to normal VMS ^C processing)

* include the following lines in the routine requiring ^C interrupts:
*	INCLUDE 'XINTERRUPT.CMN'
*
*	IF ( interrupt ) THEN ...

	EXTERNAL	CTRLC_AST
	LOGICAL		first_call
        DATA	 	first_call/.TRUE./    !kob 12/96
#ifdef unix
* ************************ UNIX ******************************
#ifdef AIX_XLF
      INTEGER SIGNAL_, old_handler, sigint
#else
      INTEGER SIGNAL, old_handler, sigint
#endif
* (see unix man signal or the include file < signal.h >)
      PARAMETER ( sigint = 2 )

      include 'ferret_cmn/xinterrupt.cmn'


*set up new defines *kob* 1/96
#ifdef AIX_XLF
# define need_underscore
# define call_handler_each_time
#endif
#ifdef solaris
# define call_handler_each_time
#endif

! "D" lines to disable disabled
!D     GOTO 888
* use newly named condition tests *kob* 1/96
#ifdef call_handler_each_time
#  ifdef need_underscore
      old_handler = SIGNAL_( 2, CTRLC_AST, -1 )    ! on AIX call every time
#  else
      old_handler = SIGNAL( 2, CTRLC_AST, -1 )    ! on solaris call every time
#  endif
#else
      IF (first_call) old_handler = SIGNAL( 2, CTRLC_AST, -1 )
#endif
!D 888 CONTINUE
#else
* ************************ VMS ******************************
	INTEGER*4	SYS$QIOW,SYS$ASSIGN,status,code,vms_mode
	INTEGER*2	input_chan

	STRUCTURE /iostat_block/
	    INTEGER*2	iostat
	    BYTE	transmit,
     .			receive,
     .			crfill,
     .			lffill,
     .			parity,
     .			zero
	END STRUCTURE
	RECORD /iostat_block/ iosb

	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'

	INCLUDE '($IODEF)'			! VMS I/O symbols
	INCLUDE '($JPIDEF)'			! for VMS mode code

* determine VMS mode (BATCH, etc.) to see if interrupts are appropriate
	CALL GET_VMS_MODE( vms_mode )
	IF ( vms_mode .NE. JPI$K_INTERACTIVE ) RETURN

	IF (first_call) THEN
* assign channel and set up QIOW structures
	   status = SYS$ASSIGN ('SYS$COMMAND',input_chan,,)
	   IF (.NOT.status) CALL LIB$SIGNAL (%VAL (status) )
	   code = IO$_SETMODE .OR. IO$M_CTRLCAST
	ENDIF

* queue an AST to handle ^C interrupt
	   status = SYS$QIOW (,
     .			      %VAL (input_chan),
     .			      %VAL (code),
     .			      IOSB,
     .			      ,,
     .			      CTRLC_AST,		! name of AST routine
     .			      ,,,,)
	   IF (.NOT.status) CALL LIB$SIGNAL (%VAL (status) )
	   IF (.NOT.  iosb.iostat) CALL LIB$SIGNAL (%VAL (iosb.iostat) )
#endif

* ************************ ALL SYSTEMS ************************
* system is ready - set flag to indicate no interrupt yet
	INTERRUPTED = .FALSE.
        first_call = .FALSE.

	RETURN
	END

***********************************************************

	SUBROUTINE CTRLC_AST

* this routine is called when ^C is entered if AST is queued

#ifdef unix
	include 'ferret_cmn/xinterrupt.cmn'	! COMMON/XINTERRUPT/
#else
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'	! COMMON/XINTERRUPT/
#endif

	interrupted = .TRUE.

	RETURN
	END	
