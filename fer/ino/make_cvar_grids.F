	SUBROUTINE MAKE_CVAR_GRIDS ( dset, status )

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
* find or make a full grid, including a time axis, for each cvar variable in 
* the given data set based on information stored in COMMON by INIT_DSET.
* results are stored COMMON/~VARIABLES/ in cvar_grid

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 -11/17/88 - based on MAKE_DS_4D_GRIDS v1.0
*			   new io library does file variables - only cvars here
* Unix/RISC port - 1/91 - cant use "/star" in comments on Unix
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include	'xvariables.cmn'
	include	'xcalc_vars.cmn'
	include	'xdset_parms.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
	INCLUDE	'FERRET_CMN:XCALC_VARS.CMN'
	INCLUDE	'FERRET_CMN:XDSET_PARMS.CMN'
#endif

* calling argument declarations
	INTEGER	dset, status

* local variable declarations
	INTEGER	TM_LENSTR1, fvar, cvar, idim, iline, slen
	CHARACTER TM_FMT*4, gname*8

* * * * CALCULATED VARIABLES
	IF ( .NOT.( dp_phil_sieg(dset) .OR. dp_cox(dset) ) ) THEN
* not available: not Philander/Siegel or Cox
	   DO 150 cvar = 1, num_calc_vars
 150	   cvar_grid( cvar, dset ) = unspecified_int4

	ELSE
* create a template for the unchanging grid stuff
	   DO 200 idim = 1, 4
 200	   grid_out_prod( idim, mgrid_buff ) = .TRUE.
	   grid_line( t_dim, mgrid_buff ) = ds_time_axis( dset )

	   DO 320 cvar = 1, num_calc_vars

* fill in the spacial axes for the grid to look for
	      DO 310 idim = 1,3
	         IF ( cvar_axes( idim, cvar ) .EQ. mnormal ) THEN
	            grid_line( idim, mgrid_buff ) = mnormal
	         ELSE
	            grid_line( idim, mgrid_buff ) =
     .			ds_basic_axes( cvar_axes(idim,cvar)-1, dset )
	         ENDIF
 310	      CONTINUE

* make a plausible name for the new grid (e.g. GPCDZ1)
	      gname = 'G' // cvar_name_code(cvar)
     .			  // TM_FMT( REAL(dset), 4, 4, slen )

* make sure name is unique
	      CALL TM_NEW_GRID_NAME( gname, gname )

* save it or use an identical one that already exists
	      CALL TM_GET_LIKE_GRID( mgrid_buff, gname,
     .				     cvar_grid(cvar,dset), status )
	      IF ( status .NE. ferr_ok ) GOTO 5000
 320	   CONTINUE
	ENDIF

* successful completion
	status = ferr_ok
 	RETURN

* error exit
 5000	RETURN
	END
