	SUBROUTINE CHECK_GRAPHICS( status )

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
* check to see if there is a graphics output ready to recieve output
* (this routine assumes that PPLUS has been started)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  3/12/90 - written to permit metafile-only output

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'ferret_cmn/xprog_state.cmn'
	include	'tmap_pplv11inc/gkscm1_inc.decl'	! with wsid
	include	'pplv11inc/GKSCM1.INC'	! with wsid
	include	'tmap_pplv11inc/pltcom_dat.decl'	! with gksopn
	include	'pplv11inc/PLTCOM.DAT'	! with gksopn
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE	'TMAP_PPLV11INC:GKSCM1_INC.DECL/LIST'	! with wsid
	INCLUDE	'PPLV11INC:GKSCM1.INC/LIST'	! with wsid
	INCLUDE	'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'	! with gksopn
	INCLUDE	'PPLV11INC:PLTCOM.DAT/LIST'	! with gksopn
#endif

* calling argument declarations
	INTEGER	status

* internal variable declarations
* ...

* only relevant for GKS output
	IF ( mode_gks ) THEN
	  IF ( wsid.EQ.0 .AND. .NOT.mode_metafile ) GOTO 5100
	ENDIF

* success
	status = ferr_ok
	RETURN

* not output device is ready
 5000	RETURN
 5100	CALL ERRMSG( ferr_state_not_set, status,
     .		     'no graphical output'//pCR//
     .		     'Use SET WINDOW or SET MODE METAFILE', *5000 )
	END
