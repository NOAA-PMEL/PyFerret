	SUBROUTINE SET_VAR( vname, status )

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
* perform a SET VARIABLE on a variable from a data set of type EZ

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200: 10/23/89
* 	 3/26/90 - renamed from SET_EZ_VAR.  Allowed user-defined vars
* V230:  7/23/92 - trap attempts to set units on user variable
*       10/22/92 - added units on uvars  (increased units string to 24 chars)
*        11/9/92 - made units and title case-sensitive
* V402: 7/95 - use PARSE_NAM_DSET instead of PARSE_NAM_DSET_GRD so the
*		grid is never determined and the underlying uvar defn need
*		not be based on variables that are valid at this moment
* V420: 10/95 - can apply SET VARIABLE/BAD=/UNIT=/TITLE= to GT and CDF files
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
* V510 5/00 *sh* - allow SET VARIABLE/BAD applied to user variables
*		 - on file variables (e.g. netCDF) interchange the 2 missing
*		   flags so that the output will inherit the manually set value
* V533 6/01 *sh* - added SET VARIABLE/NAME
*		 - increased units char buffer to 64

* v533 9/01 *acm*- Check that SET VARIABLE/NAME doesnt give a name already in the dataset
*                - and that not using a variable from a GT or TS data set
* V552: 4/03 *acm* - add SET VARIABLE/SCALEFAC/OFFSET
* v552: 4/03 *acm*  Increase length of variable names to 128 characters
* v553: 7/03 *kob* - Add SET VARIABLE/bad=NaN ability

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'

* calling argument declarations
	INTEGER		status
	CHARACTER	vname*(*)

* internal variable declarations:
	LOGICAL		VALID_GRID, TM_LEGAL_NAME, TM_HAS_STRING,
     .                  have_mods, ez, gtts, ncfile
	INTEGER		GRID_FROM_NAME, TM_LENSTR1, mod_cx, 
     .			dset, grid, cat, var, loc, TMAP_status, testvar
	REAL		bad, scalefac, offset, set_nan
	CHARACTER	grid_name*16, title*128, units*64, newname*128,
     .                  is_nan*3

* Initialize

	grid = unspecified_int4
	bad = real4_init
	scalefac = real4_init
	offset = real4_init
        have_mods = .FALSE.
        gtts = .FALSE.
        ncfile = .FALSE.
        ez = .FALSE.
	newname = char_init128
	title = char_init128
	units = char_init64
	grid_name = char_init16

* decode specified variable name of form VNAME[D=dset]
	dset = unspecified_int4			! default is global variable
	CALL PARSE_NAM_DSET( vname, cx_last, dset,
     .			     cat, var, mod_cx, status )
* ... RISC FORTRAN bug prevents "//" on char*(*) variable vname
	risc_buff = vname
	len_rbuff = LEN(vname)
	IF ( status .NE. ferr_ok ) RETURN
	IF ( var .EQ. munknown_var_name ) GOTO 5005

* must be an EZ or user-defined variable
	IF ( cat .EQ. cat_file_var ) THEN
	   ez = ds_type(dset) .EQ. pds_type_ez
!	   IF ( .NOT. ez ) GOTO 5010		! removed restriction 10/95
	ELSEIF ( cat .EQ. cat_user_var ) THEN   ! uvars bad, too, until 10/92
           ez = .FALSE.
	ELSE
	   GOTO 5015
	ENDIF

* process qualifiers given
	loc = qual_given( slash_set_var_name )
	IF ( loc .GT. 0 ) THEN

* SET VAR/NAME does not work for GT, TS datasets where sanity check is done
* in gt_find_rec.F comparing var name to the name in the dataset.

	   gtts = ( (ds_type(dset) .EQ. pds_type_gt)  .OR. 
     .              (ds_type(dset) .EQ. pds_type_ts) )
	   IF ( gtts ) GOTO 5600

	   have_mods = .TRUE.
           CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      newname, status )
           IF ( status .NE. ferr_ok ) RETURN

* check that the name isnt already in the data set.
           CALL FIND_VAR_NAME ( dset, newname, cat, testvar )
	   risc_buff = newname
	   len_rbuff = TM_LENSTR1(newname)
           IF (  testvar .NE. munknown_var_name ) GOTO 5400
	   risc_buff = vname
	   len_rbuff = LEN(vname)

	   IF ( .NOT.TM_LEGAL_NAME(newname) ) GOTO 5200
	ELSE
	   newname = char_init80
	ENDIF


* SET VAR/SCALEFAC only for NetCDF datasets

	loc = qual_given( slash_set_var_scale )
	IF ( loc .GT. 0 ) THEN

	   ncfile = (TM_HAS_STRING(ds_type(dset), 'CDF') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'MC') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'GT')  )
	   IF ( .NOT. ncfile ) GOTO 5700

	   have_mods = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   scalefac, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   scalefac = real4_init
	ENDIF

* SET VAR/OFFSET only for NetCDF datasets

	loc = qual_given( slash_set_var_offset )
	IF ( loc .GT. 0 ) THEN

	   ncfile = (TM_HAS_STRING(ds_type(dset), 'CDF') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'MC') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'GT')  )
	   IF ( .NOT. ncfile ) GOTO 5800

	   have_mods = .TRUE.
	   CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   offset, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   offset = real4_init
	ENDIF

	loc = qual_given( slash_set_var_title )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      title, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   title = char_init80
	ENDIF
	loc = qual_given( slash_set_var_unit )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      units, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   units = char_init16
	ENDIF
	loc = qual_given( slash_set_var_grid )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STRING( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      grid_name, status )
           IF ( status .NE. ferr_ok ) RETURN
	   grid = GRID_FROM_NAME( grid_name, cx_last, status )
           IF ( status .NE. ferr_ok ) RETURN
	   IF ( .NOT.VALID_GRID( grid ) ) GOTO 5500
	ELSE
	   grid = int4_init
	ENDIF
	loc = qual_given( slash_set_var_bad )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
* *kob* 7/03 - Add check for bad=NaN
	   CALL EQUAL_STRING (cmnd_buff(qual_start(loc):qual_end(loc)),
     .                         is_nan, status)
	   IF (TM_HAS_STRING(is_nan, 'NaN')) THEN
	      bad = set_nan()
	   ELSE
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   bad, status )
	      IF ( status .NE. ferr_ok ) RETURN
	   ENDIF
	ELSE
	   bad = real4_init
	ENDIF

* note on EZ vars:
* modifying the variable info invalidates data in memory
*        the purging of memory variables below is unnecessarily strict in both
*	 the uvar and fvar cases.  FVAR's can be modified in memory to change
*	 mr_grid, units, etc.  They need to be deleted only if they were on 
* 	 non-standard grids or units.  The UVAR's need to be purged only if
*	 they depend on the FVAR being changed)
	IF ( ez ) THEN
	   IF ( have_mods ) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	      CALL EZ_MOD_VARS( dset,var,
     .			        newname,title,units,grid,bad,tmap_status )
	      IF ( tmap_status .NE. merr_ok ) THEN
	         status = ferr_TMAP_error
	         RETURN
	      ENDIF
	   ENDIF
	ELSEIF ( cat .EQ. cat_file_var ) THEN
* ... other file variables (10/95)
	   IF ( grid .NE. int4_init   ) GOTO 5112
	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	      ds_missing_flag(var) = ds_bad_flag(var)
	      ds_bad_flag    (var) = bad
	   ENDIF
*	   IF (newname.NE. char_init16 ) ds_var_code ( var ) = newname
	   IF (newname.NE. char_init16 )
     .         CALL string_array_modify( ds_var_code_head, var,
     .                                   newname, LEN(newname))
	   IF ( units .NE. char_init16 ) ds_var_units( var ) = units
	   IF ( title .NE. char_init80 ) ds_var_title( var ) = title

           IF (scalefac .NE. real4_init .AND. ncfile) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
              ds_var_scale(var) = scalefac
              ds_var_scaleit(var)       = .TRUE.
           ENDIF
           IF (offset .NE. real4_init .AND. ncfile) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
              ds_var_off(var)   = offset
              ds_var_scaleit(var)       = .TRUE.
           ENDIF
	ELSE
* ... user-defined var
	   IF ( grid .NE. int4_init   ) GOTO 5110
*	   IF (newname.NE. char_init16 ) uvar_name_code(var)= newname
	   IF (newname.NE. char_init16 ) 
     .         CALL string_array_modify(uvar_name_code_head,
     .                                  var,
     .                                  newname,
     .                                  LEN(newname))
	   IF ( units .NE. char_init16 ) uvar_units( var )  = units
	   IF ( title .NE. char_init80 ) uvar_title( var )  = title
	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_ALL_UVARS	 ! delete invalidated uvars
	      uvar_bad_data(var) = bad
	   ENDIF

	ENDIF

* successful completion
	status = ferr_ok
	RETURN

* error exits
 5000	RETURN
 5005	CALL ERRMSG( ferr_unknown_variable, status,
     .		    '(data set ok ?): '//risc_buff(:len_rbuff), *5000 )
! 5010	CALL ERRMSG( ferr_invalid_command, status,
!     .		risc_buff(:len_rbuff)//' is from a self-documented data set'
!     .		//pCR//'Can only SET VARIABLE on variable from an EZ data set',
!     .							 *5000 )
 5015	CALL ERRMSG( ferr_invalid_command, status,
     .		'Cannot SET VARIABLE on '//risc_buff(:len_rbuff)//pCR//
     .		'Only on variables from "EZ" data sets',
     .							 *5000 )
!     .		'Only on user-defined and variables from "EZ" data sets',
 5100	CALL ERRMSG( ferr_erreq, status,
     .		cmnd_buff(qual_start(loc):qual_end(loc)), *5000 )
 5110   CALL ERRMSG( ferr_invalid_command, status,
     .               'cannot specify grid for '//risc_buff(:len_rbuff)//
     .		      pCR//
     .               'Grid for a user-defined variable is implied '//
     .               'by its definition', *5000 )
 5112   CALL ERRMSG( ferr_invalid_command, status,
     .               'cannot specify grid for '//risc_buff(:len_rbuff),
     .               						*5000 )
 5200	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal new name given for '
     .			//risc_buff(:len_rbuff),
     .               						*5000 )
 5400	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal new name.  Variable '
     .			//risc_buff(:len_rbuff)//' exists in dataset ',
     .                                                          *5000 )
 5500	CALL ERRMSG( ferr_grid_definition, status, grid_name, *5000 )
 5600	CALL ERRMSG( ferr_invalid_command, status,
     .               'Cannot SET VARIABLE/NAME= with GT or TS datasets',
     .               						*5000 )
 5700	CALL ERRMSG( ferr_invalid_command, status,
     .           'Can SET VARIABLE/SCALEFAC= only with NetCDF datasets',
     .               						*5000 )
 5800	CALL ERRMSG( ferr_invalid_command, status,
     .           'Can SET VARIABLE/OFFSET= only with NetCDF datasets',
     .               						*5000 )
	END
