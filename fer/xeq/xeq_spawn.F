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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/2/88
* v200: Unix/RISC port - 3/12/91 - coach for "^z" on Unix
* V230 6/92 - allow "system" calls for unix
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_state.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_STATE.CMN'
#endif

# ifdef unix
	IF ( num_args .GT. 0 ) THEN
	   call system( cmnd_buff(arg_start(1):len_cmnd) )
	ELSE
* user can get choice of shells with "SPAWN csh", or whatever.
	   CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .    	' Use "^Z" for shell prompt.  Type "fg" to return.', 0)
	   CALL SPLIT_LIST(pttmode_help, ttout_lun,
     .    	' Or type "SPAWN csh" (or shell of your choice) '
     .    	//'and "exit" to return', 0)
      ENDIF
#else
* VMS systems ...
* open PPLUS if it isn't yet
	IF ( .NOT.pplus_started ) THEN
	   CALL START_PPLUS
	ENDIF

* simply use the PPLUS SPAWN command
	CALL PPLCMD (' ',' ',0,'SPAWN',1,1)
#endif

	RETURN

	END

