	SUBROUTINE ALL_1_ARG_EXPR

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

* This routine is a clone of ALL_1_ARG, but altered so that if there are
* overall surrounding quotation marks, they *are* passed on, so
*	yes? LIST "myvar"
* is seen as the string "myvar", rather than the value of the variable myvar

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V694: 2/2/2015

* calling argument declarations:

* internal variable declarations:
	INTEGER i

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xprog_state.cmn'

* reset the command common to look like a single big argument
* unlike ALL_1_ARG, do this even if there is only a single arg
	IF (num_args .EQ. 0) RETURN

	arg_end(1) = arg_end(num_args)
	num_args = 1

* protect against lost " from start 
	i = arg_start(1) - 1
	IF ( cmnd_buff(i:i) .EQ. '"' ) arg_start(1) = i

* protect against lost " from end
	i = arg_end(1) + 1
	IF ( i .LE. len_cmnd ) THEN
	   IF ( cmnd_buff(i:i) .EQ. '"' ) arg_end(1) = i
	ENDIF

* protect against lost _DQ_ from start
	i = arg_start(1) - 4
	IF (i .GE. 1) THEN
	   IF ( cmnd_buff(i:i+3) .EQ. '_DQ_' ) arg_start(1) = i
	ENDIF

* protect against lost _DQ_ from end
	i = arg_end(1) + 4
	IF ( i .LE. len_cmnd ) THEN
	   IF ( cmnd_buff(i-3:i) .EQ. '_DQ_' ) arg_end(1) = i
	ENDIF

	RETURN
	END
