	PROGRAM FERRET

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
* TMAP interactive data analysis program

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* FERRET program history:
* initially tailored to output format and content of the Philander/Seigel 
* model from GFDL
* revision 0.0  - 4/3/86
* revision 1.0 - 11/17/86 - first "official" release
* revision 2.0 - 10/23/87 - "official" release
* revision 2.01(temporary) - 10/23/87 - smaller memory size, bug fixes,
*			2 typos in XVARIABLES_DATA, ^C added, ZT planes added
* revision 2.02 - ?????
* revision 2.10 - 5/6/88 - "final" release - /NODEBUG version
* FERRET 1.00     - 6/10/88 - rename of GFDL 2.10
* FERRET 1.10     -  8/2/88 - numerous bug fixes and enhancements
* FERRET 1.20     - 2/17/89 - numerous bug fixes and enhancements
* FERRET 1.21     - 4/19/89 - minor bug fixes
* FERRET 2.00	  - 5/??/89 - internal re-write: 4D grids and "object oriented"
*			      transformations
* FERRET 3.00     - 1/29/93 - revision 2.2-->2.3 changes became so extensive
*                             and prolonged it made sense to rename to V3.0
* FERRET 3.10     - 4/94 - official release using XGKS
* FERRET 3.11     - 5/94 - added FILE/ORDER=/FORMAT=STREAM
* FERRET 3.12     - 5/94 - restructured to be "dynamic memory" (C main routine)
*			   former MAIN became FERRET_DISPATCH routine
* FERRET 3.13     - 7/94 - relink of Solaris version
*                          (using IBM-portable TMAP libs)
*|*|*|*|*|*|*|*|*|*|*|*|
* FERRET 4.0     - 7/94 - using a C main program with dynamic memory
*Note: this module is a FORTRAN MAIN for FERRET to asist with porting
*	The full fully functional main program is written in c

* 12/94 - modified ferret_dispatch arguments

#ifdef unix
	include 'tmap_errors.parm'   ! merr_ok
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'gui.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
#else
	INCLUDE 'TMAP_ERRORS.PARM'   ! MERR_OK
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:GUI.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
#endif

	EXTERNAL CTRLC_AST

* local parameter declarations
	INTEGER	pnflags, pnchars
	PARAMETER (pnflags = 10,
     .		   pnchars = 500 )

* local variable declarations
	LOGICAL do_init
	INTEGER TM_LENSTR1, status, rtn_flags(pnflags), i, hlen,
     .		nrtn_chars, nrtn_errlines
	CHARACTER*128  home_path
#ifdef NEED_BYTE
        BYTE      init_command(24), rtn_chars(pnchars)
#else
        INTEGER*1 init_command(24), rtn_chars(pnchars)
#endif


* temporarily we simply declare the "memory" array right here
	REAL memory(pmem_blk_size,pmax_mem_blks)

* initialize stuff: keyboard, todays date, grids, GFDL terms, PPL brain
	CALL INITIALIZE

* turn on ^C interrupts
	CALL SET_CTRL_C( CTRLC_AST )

*  prepare appropriate console input state and open the output journal file
	CALL INIT_JOURNAL( status )

* program name and revision number
	CALL DISTRIB_MSG

* initialize size and shape of memory and linked lists
	CALL INIT_MEMORY( pmem_blk_size, pmax_mem_blks )

* ==>> place FERRET under GUI control here  <<==
!	CALL FERRET_DISPATCH( memory, '\SET MODE GUI', rtn_buff )
!	check status and issue messages if errors ...

* set up to execute $HOME/.ferret if it exists
* The command to be passed must be set up as a null terminated, c-style string
	home_path = ' '
	CALL GETENV('HOME',home_path)
	hlen = TM_LENSTR1(home_path)
	INQUIRE( FILE = home_path(:hlen)//'/.ferret', EXIST = do_init )
	IF ( .NOT. do_init ) THEN
	   init_command(1) = 0
	ELSE
	   home_path = '\GO "$HOME/.ferret" '
	   DO 5 i = 1, 20
 5	   init_command(i) = ICHAR( home_path(i:i) )
	   init_command(21) = 0
	ENDIF

* Now over to FERRET
* Note: if not under GUI control we will not normally exit FERRET_DISPATCH
*	  until we are ready to exit FERRET
! 10	CALL FERRET_DISPATCH( memory, init_command, rtn_buff )
 10	CALL FERRET_DISPATCH( memory, init_command, rtn_flags, pnflags,
     .				rtn_chars, nrtn_chars, nrtn_errlines )


* debugging flow control checks
	init_command(1) = 0
	IF ( rtn_flags(frtn_action) .EQ. factn_mem_reconfigure ) THEN
	   WRITE (6,*) 'Cannot reconfigure memory from FORTRAN:',
     .						rtn_flags(frtn_idata1)
	   GOTO 10
	ELSEIF ( rtn_flags(frtn_action) .NE. factn_exit ) THEN
	   WRITE ( 6,* ) 'Would-be GUI in control ...'
	   GOTO 10
	ELSE
	   CALL EXIT(0)
	ENDIF

	END


