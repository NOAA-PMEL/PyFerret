	SUBROUTINE MAKE_PSEUDO_VAR ( cx, idim, pvar, mr, lo_ss, hi_ss )

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
* create a "pseudo-variable" - numerical grid of data based solely on the
* underlying grid (not involving external data)

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/2/87
* V200:  6/20/89 - 4D-symmetrical version
* V530   1/01 - *sh* - added XBOXLO, XBOXHI and family ...

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include 'xvariables.cmn'
	include 'xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx, mr, idim, lo_ss, hi_ss
	REAL	pvar( lo_ss:hi_ss )

* internal variable declarations:
	INTEGER	grid, pv, i
	REAL	BOX_SIZE
	REAL*8	TM_WORLD

* initialize

* defining grid for output data
	grid = cx_grid( cx )

	pv = (mr_variable(mr)-1) / 4  + 1
	IF ( pv .GT. 5 )  STOP 'MK_PSEUDO_VAR'	! diag

	GOTO (  100, 200, 300, 400, 500 ) pv

* "I", "J", "K", "L"
 100	DO 110 i = lo_ss, hi_ss
 110	pvar( i ) = i
	RETURN

* "X", "Y", "Z", "T"
 200	DO 210 i = lo_ss, hi_ss
 210	pvar( i ) = TM_WORLD( i,grid,idim,box_middle )
	RETURN

* "XBOX", "YBOX", "ZBOX", "TBOX"
 300	DO 310 i = lo_ss, hi_ss
 310	pvar( i ) = BOX_SIZE( i, grid, idim )
	RETURN

* "XBOX", "YBOX", "ZBOX", "TBOX"
 400	DO 410 i = lo_ss, hi_ss
 410	pvar( i ) =  TM_WORLD( i,grid,idim,box_lo_lim )
	RETURN

* "XBOX", "YBOX", "ZBOX", "TBOX"
 500	DO 510 i = lo_ss, hi_ss
 510	pvar( i ) =  TM_WORLD( i,grid,idim,box_hi_lim )
	RETURN

	END
