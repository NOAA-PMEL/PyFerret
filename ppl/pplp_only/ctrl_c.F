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

* argument: CTRLC_AST
*	name of routine to be called if operator hits ^C
*	(or call with argument %VAL(0) to return to normal VMS ^C processing)

* include the following lines in the routine requiring ^C interrupts:
*	INCLUDE 'XINTERRUPT.CMN'
*
*	IF ( interrupt ) THEN ...

	EXTERNAL	CTRLC_AST
CC	Fix for fortran 90 -- initialize in data stmt
	LOGICAL		first_call
	DATA		first_call/.TRUE./

#ifdef unix
* ************************ UNIX ******************************
      INTEGER SIGNAL, old_handler, sigint
* (see unix man signal or the include file < signal.h >)
      PARAMETER ( sigint = 2 )

C      include 'xinterrupt.cmn'
C      xinterrupt.cmn follows *jd* 2.4.93
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	COMMON /XINTERRUPT/ interrupted

	LOGICAL interrupted
C+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
C Comment out the D lines FORTRAN_90 *jd* 1.21.97
C "D" lines to disable disabled
CD     GOTO 888
      IF (first_call) old_handler = SIGNAL( 2, CTRLC_AST, -1 )
CD 888 CONTINUE
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

	INCLUDE 'XINTERRUPT.CMN'

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
!	include 'xinterrupt.cmn'	! COMMON/XINTERRUPT/
!      xinterrupt.cmn follows *jd* 2.4.93
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	COMMON /XINTERRUPT/ interrupted

	LOGICAL interrupted
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#else
	INCLUDE 'XINTERRUPT.CMN'	! COMMON/XINTERRUPT/
#endif

	interrupted = .TRUE.

	RETURN
	END	
