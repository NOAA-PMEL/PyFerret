	SUBROUTINE POS_LAB( ss, cx, idim, ndec, string, slen )

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
* create a string of the form (e.g.) "I:22  X:130.5E"
* if grid is unknown then "I:22", alone
* if subscript is unknown then "I:*"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   ???? 
*	 7/10/90 - added checks for invalid grid and ss

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* calling argument declarations:
	INTEGER		ss, cx, idim, ndec, slen
	CHARACTER*(*)	string
	
* local variable declarations:
	INTEGER		TM_LENSTR1, grid
	CHARACTER*4	TM_FMT
	REAL*8		TM_WORLD

* initialize string with subscript (e.g.) "I:22 " or "I:*"
	IF ( ss .NE. unspecified_int4 ) THEN
	   string = ss_dim_name(idim) // ':' // TM_FMT( FLOAT(ss),1,4,slen )
	   slen = slen + 2
	ELSE
	   string = ss_dim_name(idim) // ':*'
	   slen = 3
	   RETURN
	ENDIF

* done with pathological cases
	IF ( cx .GT. 0 ) THEN
	   grid = cx_grid( cx )
	ELSE
	   RETURN
	ENDIF

* add on world coord axis name (e.g.) "I:22  X:"
	string = string(:slen+1) // ww_dim_name(idim) // ':'

* add formatted world coordinate position
	CALL TRANSLATE_TO_WORLD
     .	   ( TM_WORLD( ss, grid, idim, box_middle ),
     .	     idim, grid, ndec, string(slen+7: ) )

* compute string length
	slen = TM_LENSTR1( string )

	RETURN
	END
