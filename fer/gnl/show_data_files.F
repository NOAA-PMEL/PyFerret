	SUBROUTINE SHOW_DATA_FILES( lun, dset )

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
* revision 0.0 - 8/15/88
* V230: 6/92 - elaborated on kludge to locate time axis for data set
*            - use the first variable that HAS a time axis
*              (not clear why ds_time_axis cannot be used ??)
* V301: 4/94 - show file for T-independent data set, too
*	       but not for a netCDF data set
* V320:12/94 - increased the output field size for /COLUMNS=xxxx
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#	include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xprog_state.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER	lun, dset
	
* local variable declarations:
	INTEGER		KNOWN_GRID, TM_LENSTR1,
     .			ivar, grid, istep, ndec, status, lskip, slen
	CHARACTER	LEFINT*8, lo_date*20, hi_date*20, skip*8

* special processing of EZ data sets
	IF ( ds_type(dset) .EQ. pds_type_ez ) THEN
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /FORMAT = '//ds_parm_text(1,dset), 0)
           DO 10 istep = 1,maxstepfiles
 10	   IF (sf_setnum(istep) .EQ. dset) GOTO 20
	   CALL ERRMSG( ferr_internal, status, 'EZ_ err', *5000 )
 20	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /SKIP = '//LEFINT( sf_skip(istep), lskip ), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /COLUMNS = '//LEFINT( ds_ntuple(dset), lskip ), 0)
 5000	   RETURN

* nothing to show for netCDF files (4/94)
        ELSEIF (ds_type(dset) .EQ. 'CDF'
     .    .OR.  ds_type(dset) .EQ. 'ECDF') THEN
	   RETURN
	ENDIF

* !kludge - get the time axis from a variable in this data set
	DO 100 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset ) THEN
	      grid = KNOWN_GRID( dset, cat_file_var, ivar )
              IF ( grid_line(t_dim,grid) .NE. mnormal ) GOTO 200
	   ENDIF
 100	CONTINUE

******* 4/94 addition ******
* a time-independent data set
* list files belonging to this data set (should be one, only) 
	DO 110 istep = 1, maxstepfiles
	    IF ( sf_setnum(istep) .EQ. dset ) CALL SPLIT_LIST
     .		(pttmode_explct, lun, ' time-independent data file: '
     .		//sf_name(istep), 0 )
 110	CONTINUE
******* end of 4/94 addition ******
	RETURN

* a time-dependent data set:
* write heading line for step file list
 200	CALL SPLIT_LIST(pttmode_explct, lun,
     .		'          time-dependent data files:', 36)

* output string precision (from SET MODE CALENDAR)
           ndec = ax_dec_pt( t_dim )

* see which step files belong with this data set 
	DO 300 istep = 1, maxstepfiles
	    IF ( sf_setnum(istep) .EQ. dset ) THEN
	      CALL TRANSLATE_TO_WORLD(	sf_1step(istep), t_dim, grid,
     .					ndec, lo_date )
	      CALL TRANSLATE_TO_WORLD(	sf_lastep(istep), t_dim, grid,
     .					ndec, hi_date )
	      slen = TM_LENSTR1(hi_date)
	      CALL SPLIT_LIST(pttmode_explct, lun,
     .		lo_date(:slen)//' -> '//hi_date(:slen)//'    '
     .		//sf_name(istep), 0 )
	   ENDIF
 300	CONTINUE

	RETURN
	END

