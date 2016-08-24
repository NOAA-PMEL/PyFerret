	SUBROUTINE XEQ_SPAWN

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
* SPAWN a subprocess to VMS DCL level

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/2/88
* v200: Unix/RISC port - 3/12/91 - coach for "^z" on Unix
* V230 6/92 - allow "system" calls for unix
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* 10.28.98 *js* Disable command if secure mode set
* V500:  1/99 - correct the handing of qutes as in SPAWN "date"
*		with call to ALL_1_ARG
* V606  8/07 *acm* Send informational and error messages that are returned from 
*                  commands via SPLIT_LIST to std error rather than std out.
*       *acm* 3/12 cleanup ifdefs and unnecessary include files
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
*    *acm* 7/17. Test a status flag on the system call (see ticket 2453). Results 
*                seem inconsistent, make it a warning

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'
	include 'xplot_state.cmn'

	LOGICAL IS_SECURE
	INTEGER	status, slen

	CALl ALL_1_ARG
	IF ( num_args .GT. 0 ) THEN
	   if (IS_SECURE()) then
	      call SPLIT_LIST(pttmode_help, err_lun,
     1	        'This command is not allowed.', 0)
	   else
*             Flush Fortran stdout and stderr so that any output from
*             the system command will appear in the appropriate place.
              CALL FLUSH(ttout_lun)
              CALL FLUSH(err_lun)
	      CALL SYSTEM( cmnd_buff(arg_start(1):arg_end(1)),status )
	      IF (status .NE. 0) CALL WARN (
     .           'Shell command returned failure flag: '//
     .            cmnd_buff(arg_start(1):arg_end(1)) )

* To handle a nonzero return as an ERROR, comment out the WARN just above, and do this.
c	      IF (status .NE. 0) GOTO 5100   

	   end if
	ELSE
* user can get choice of shells with "SPAWN csh", or whatever.
	   CALL SPLIT_LIST(pttmode_help, err_lun,
     .    	' Use "^Z" for shell prompt.  Type "fg" to return.', 0)
	   CALL SPLIT_LIST(pttmode_help, err_lun,
     .    	' Or type "SPAWN csh" (or shell of your choice) '
     .    	//'and "exit" to return', 0)
	ENDIF

5000	CONTINUE
	RETURN

5100	CONTINUE


* The return is the return flag from the command, which might be non-zero
* if they look for a file that's non-existent. 
* (Should this really be an ERROR!?  I'm thinking not) 
*   sp rm nothing.dat >> returns 256. 
*   sp ls nofile.dat  >> returns 512
*   sp cat file.dat | grep "this text not in the file"  >> returns 256
*   sp notAcommand >> returns 32512

	CALL ERRMSG(ferr_sys_return, status, 
     .            'SPAWN '//cmnd_buff(arg_start(1):arg_end(1)), *5000)

	END

