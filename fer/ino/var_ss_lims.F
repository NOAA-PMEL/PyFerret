	SUBROUTINE VAR_SS_LIMS( idim, cx, lo_ss, hi_ss )

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/19/89 - created
*	10/25/89 - streamlined determining file variable number
* V300: 10/28/93 - apply limits to pseudo-variables, too, if grid is suitable
*	note: this addition was made YEARS after the logic here was developed
*	      there are uncertainties whether special circumstances in FERRET
*	      may lead to problem behavior

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xfr_grid.cmn'	! 10/93
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	include	'FERRET_CMN:XFR_GRID.CMN'
#endif

* calling argument declarations
	INTEGER idim, cx, lo_ss, hi_ss

* internal variable declarations
	INTEGER dset, cat, var, grid, line, status
	INTEGER TM_GET_GRIDNUM		! 10/93

* initialize
	dset = cx_data_set(cx)
	var  = cx_variable(cx)
	cat  = cx_category(cx)
	grid = cx_grid(cx)
	IF ( grid .NE. unspecified_int4 ) line = grid_line( idim, grid )

* pseudo-variables - added 10/93
* (must be a known axis and not be ABSTRACT)
	IF ( cat  .EQ. cat_pseudo_var
     . .AND. grid .NE. unspecified_int4
     . .AND. line .NE. unspecified_int4
     . .AND. grid .NE. TM_GET_GRIDNUM('ABSTRACT')  ) THEN
	   lo_ss = 1
	   hi_ss = line_dim( line )
	   RETURN
	ENDIF
* ... end of 10/93 additions

* cannot return answer under several conditions
	IF ( cat  .NE. cat_file_var
     .  .OR. dset .EQ. unspecified_int4
     .  .OR. line .EQ. mnormal
     .  .OR. cx_unstand_grid(cx)	 ) THEN
	   lo_ss = unspecified_int4
	   hi_ss = unspecified_int4
	   RETURN
	ENDIF

* return subscript limits
 110	lo_ss = ds_grid_start(idim, var)
	hi_ss = ds_grid_end  (idim, var)
 1000	RETURN
	END
