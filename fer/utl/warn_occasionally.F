	SUBROUTINE WARN_OCCASIONALLY ( string, ncycle, cookie )

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
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* display a warning message to the operator, but do not display it again
* until it has been requested at least ncycle times.
* each message must be associated with a unique, arbitrary cookie.

* V690 12/13 *sh*
** 10/2022 *acm* Code cleanup: remove references to obsolete gui interface

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'xprog_state.cmn'

* calling argument declarations:
	INTEGER		ncycle, cookie
	CHARACTER*(*)	string

* internal variable declarations:
	INTEGER i

* private COMMON storage used only by this routine
	INTEGER	pn_uniq_msgs
	PARAMETER (pn_uniq_msgs = 10)
	INTEGER	nused, counter(pn_uniq_msgs), cookies(pn_uniq_msgs)
	COMMON / XWARN_OCCASIONALLY / counter
	DATA nused / 0 /
	DATA counter /	pn_uniq_msgs*0 /

* have we seen this cookie before?
	DO 100 i = 1, nused
	   IF (cookies(i) .EQ. cookie) THEN
* ... a repeat request for this message
	      IF (MOD(counter(i),ncycle) .EQ. 0) THEN
	         GOTO 500
	      ELSE
	         counter(i) = counter(i) + 1
	      ENDIF
	      RETURN
	   ENDIF
 100	CONTINUE

* this is a first time request -- cllaim a spot
	nused = nused + 1
	IF (nused .GT. pn_uniq_msgs) THEN
	   CALL WARN('crptn: warn_occasionally ovfl')	
	   RETURN
	ENDIF
	cookies(nused) = cookie

* issue the warning and start the count-up cycle
 500	CALL WARN(string)
	CALL WRITE_MULTI(err_lun,
     .	   "           (This note will be repeated occasionally.)")
	counter(i) = 1
	RETURN
	END
