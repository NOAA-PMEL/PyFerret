      SUBROUTINE CK_GKS_ACTIVE( * )

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
* check if GKS is avtive - if not declare error and take abnormal return

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 12/20/89

#ifdef unix
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'pltcom_dat.decl'   ! with gksopn
	include 'PLTCOM.DAT'   ! with gksopn
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL/LIST'   ! with gksopn
	INCLUDE 'PPLV11INC:PLTCOM.DAT/LIST'   ! with gksopn
#endif

	external errmsg

* calling argument declarations:
*	...

* internal variable declarations:
	INTEGER	state, status

* irrelevant for PLTYPE .NE. 3 or 4
	IF ( .NOT.gksopn ) RETURN

* get GKS operating state
	CALL GQOPS( state )

* states:
*	0 - GKS closed
*	1 - GKS open
*	2 - At least 1 workstation is open
*	3 - At least 1 workstation is active
*	4 - a segment is open

	IF ( state .GE. 3 ) RETURN
	   
* GKS isn't ready !!
	CALL ERRMSG( ferr_state_not_set, status,
     .		'graphical output device isnt ready', *5000 )
 5000	RETURN 1

	END
