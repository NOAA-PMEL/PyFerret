	INTEGER FUNCTION GRID_FROM_NAME(lc_name, dflt_cx, status)

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
* given a name which may be the name of a grid or the name of a variable
* with possible data set and grid name modifiers return a grid number

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 7/29/88
* V200:  4/27/90 - bug fix: status not set on successful return
* Unix/RISC port - 11/1/91: bug fix: status not checked returning from
*                           PARSE_NAM_DSET_GRD.  "SHOW GRID T" blew up.
* V300: 2/93 - return unspecified if error exit (caught by f77 3.2)
* V500 7/98 *sh* - allow case-sensitive variable names in single quotes 
*      3/99 *sh* - disallow implicit grids (e.g. "Z[gz=zax]")

#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER		dflt_cx, status
	CHARACTER*(*)	lc_name

* local variable declarations:
	LOGICAL		made_impl_grid
	INTEGER		STR_UPCASE, TM_GET_GRIDNUM, TM_LENSTR1,
     .			vax_code, iset, grid, category, variable
	CHARACTER	name*32

* change it to upper case
	vax_code = STR_UPCASE( name, lc_name )

* try to identify it first as a grid name
	grid = TM_GET_GRIDNUM( name )
	IF ( grid .NE. unspecified_int4 ) GOTO 1000
	
* well then, identify the variable named and get its grid
	CALL PARSE_NAM_DSET_GRD( lc_name, dflt_cx,
     .	  iset, grid, category, variable, made_impl_grid, status )
	IF (made_impl_grid) THEN		!3/99
	   CALL DEALLO_GRID( status )
	   GOTO 5100
	ENDIF
        IF ( status .NE. ferr_ok ) GOTO 5000       ! 11/91 *sh*
	IF ( variable .EQ. munknown_var_name ) CALL ERRMSG
     .       ( ferr_unknown_grid, status,
     .             lc_name(:TM_LENSTR1(lc_name)),*5000 )
	
* success
 1000	GRID_FROM_NAME = grid
	status = ferr_ok
	RETURN

* error exit(s)
 5000	GRID_FROM_NAME = unspecified_int4     ! added 2/93
  	RETURN
 5100	CALL ERRMSG(ferr_invalid_command, status,
     .			'disallowed usage of '//lc_name, *5000)

	END
