	SUBROUTINE ALL_1_ARG

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
