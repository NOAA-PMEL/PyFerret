	SUBROUTINE SEND_PLTYPE( meta )

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
* send the PLTYPE command to PPLUS including workstation type and metafile
* information

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* V200:   1/5/90
* V230:  2/11/92 - (SUN port) fixed bug that "WS=DEFAULT" was sent to PPLUS
*        11/9/92 - eliminate device dependence (no WS=, no META=)
* V310:  12/4/94 - use mode_GKS to determine what PLTYPE to send

* calling argument declarations:
	LOGICAL meta

* local veriable declarations:
	INTEGER	TM_LENSTR1

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include 'ferret_cmn/xplot_setup.cmn'
	include 'ferret_cmn/xgks_devices.cmn'
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'FERRET_CMN:XPLOT_SETUP.CMN'
	INCLUDE 'FERRET_CMN:XGKS_DEVICES.CMN'
#endif

* command format: "PLTYPE 3 META"
	IF ( mode_gks ) THEN
	   ppl_buff = 'PLTYPE 3'
	ELSE
	   ppl_buff = 'PLTYPE 1'	!  4/94
	ENDIF

* append META=device
	IF ( meta ) ppl_buff = ppl_buff(:8)//' META'

* send it
	CALL PPLCMD ( from, line, 0, ppl_buff, 1, 1 )

	RETURN
	END
