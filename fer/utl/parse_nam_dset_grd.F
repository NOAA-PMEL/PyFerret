	SUBROUTINE PARSE_NAM_DSET_GRD( full_var_name, default_cx,
     .				       dset, grid, cat, var, status )

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
* given a full variable specification of the form:
*			(eg.) TEMP[D=n,G=gname] (with optional region info)
* return context information describing variable, data set and  grid
* ... if necessary determine the grid in the process

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 7/27/88 - based on GET_VAR_CONTEXT and PARSE_VAR_NAME
* revision 0.1 - 3/14/89 - check to see if default data set is known
* V200:   8/9/89 - eliminated unknown data set error
* V402: 7/95 - Allow nested [] of the form vname1[G=vname2[D=dset]]
*	     - Guts of routine extracted to PARSE_NAM_DSET
	
#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, dset, grid, cat, var, status
	CHARACTER*(*)	full_var_name

* internal variable declarations:
	INTEGER		IGRID, mods_cx

* most of the work extracted to this routine (7/95)
	dset = unspecified_int4		! no prejudices ??? (7/95)
	CALL PARSE_NAM_DSET( full_var_name, default_cx,
     .			     dset, cat, var, mods_cx, status )

* determine the defining grid
* (although mods_cx has been deallocated nothing has over-written it yet)
	IF ( var .NE. munknown_var_name ) THEN
	   IF ( cx_unstand_grid( mods_cx ) ) THEN
	      grid = cx_grid( mods_cx )
	   ELSE
	      grid = IGRID( dset, cat, var, status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
	   ENDIF
	ENDIF

* error exit
 5000	RETURN
	END
