	SUBROUTINE SET_CTRL_C (CTRLC_AST)

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
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
* 1/03 *kob* - g77 port - g77 needs to call SIGNAL with only two arguments

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
#if defined AIX_XLF 
      INTEGER SIGNAL_, old_handler, sigint
#else
      INTEGER SIGNAL, old_handler, sigint
#endif
* (see unix man signal or the include file < signal.h >)
      PARAMETER ( sigint = 2 )

      include 'xinterrupt.cmn'


*set up new defines *kob* 1/96
#ifdef AIX_XLF 
# define call_handler_each_time
#endif
#ifdef solaris
# define call_handler_each_time
#endif

! "D" lines to disable disabled
!D     GOTO 888
* use newly named condition tests *kob* 1/96
#ifdef call_handler_each_time
#  ifdef NEED_SIGNAL_UNDERSCORE
      old_handler = SIGNAL_( 2, CTRLC_AST, -1 )    ! on AIX call every time
#  elif defined G77_SIGNAL
      old_handler = SIGNAL( 2, CTRLC_AST)    !! on solaris call every time
#  else
      old_handler = SIGNAL( 2, CTRLC_AST, -1 )    ! on solaris call every time

#  endif
#else
#  ifdef NEED_SIGNAL_UNDERSCORE
      IF (first_call) old_handler = SIGNAL_( 2, CTRLC_AST, -1 )
#  elif defined G77_SIGNAL
      old_handler = SIGNAL( 2, CTRLC_AST)    
#  else 
      IF (first_call) old_handler = SIGNAL( 2, CTRLC_AST, -1 )
#  endif	
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
	include 'xinterrupt.cmn'	! COMMON/XINTERRUPT/
#else
	INCLUDE 'FERRET_CMN:XINTERRUPT.CMN'	! COMMON/XINTERRUPT/
#endif

	interrupted = .TRUE.

	RETURN
	END	
