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
* V600:11/05 *acm* - issue error for SET VAR/SCALE/OFFSET if dset = -111 
* V600: 2/06 *acm* - For attribute-access additions, add new SET VAR/OUTTYPE=
* V600  4/06 *acm* - additional declarations found by Solaris compiler
* v600  4/06 *acm* - Change the call to set_nan from a float function to void, passing
*                    back the result as an argument. Needed for port to 64-bit 
* V601 10/06 *acm* - When units, title, or missing is set, change the corresponding
*                    value or string in the attribute linked-list structure.
* V608 11/07 *acm* - If doing SET VAR/UNITS, /TITLE, /BAD on a a user variable,
*                    and that attribute was not previously set, add it.

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'slash.parm'
        include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
        include 'netcdf.inc'

* calling argument declarations
	INTEGER		status
	CHARACTER	vname*(*)

* internal variable declarations:
	LOGICAL		VALID_GRID, TM_LEGAL_NAME, TM_HAS_STRING,
     .                  have_mods, ez, gtts, ncfile, coordvar
	INTEGER		GRID_FROM_NAME, TM_LENSTR1, STR_UPCASE, mod_cx,
     .			i, dset, grid, cat, var, loc, TMAP_status, 
     .			testvar, outtype_flag, varid, attid, attype, attlen, 
     .			attoutflag, vartype, nvdims, nvatts, vdims(8), 
     .			varoutflag, cx, idum
	REAL		bad, scalefac, offset, val, dummy
	CHARACTER	grid_name*16, title*128, units*128, newname*128,
     .                  is_nan*3, flag*10, typflag*10, varattname*512,
     .                  varname*128, attname*128, buf*2, newname_lc*128

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

* May be an EZ or user-defined variable
	IF ( cat .EQ. cat_file_var ) THEN
	   ez = ds_type(dset) .EQ. pds_type_ez
!	   IF ( .NOT. ez ) GOTO 5010		! removed restriction 10/95
	ELSEIF ( cat .EQ. cat_user_var ) THEN   ! uvars bad, too, until 10/92
           ez = .FALSE.
	ELSE
	   GOTO 5015
	ENDIF

* process qualifiers given

* SET VAR/NAME=

	loc = qual_given( slash_set_var_name )
	IF ( loc .GT. 0 ) THEN

* SET VAR/NAME does not work for GT, TS datasets where sanity check is done
* in gt_find_rec.F comparing var name to the name in the dataset.

	   gtts = ( (ds_type(dset) .EQ. pds_type_gt)  .OR. 
     .              (ds_type(dset) .EQ. pds_type_ts) )
	   IF ( gtts ) GOTO 5600

	   have_mods = .TRUE.
           CALL EQUAL_STR_LC 
     .             ( cmnd_buff(qual_start(loc):qual_end(loc)),
     .               newname_lc, status )
           IF ( status .NE. ferr_ok ) RETURN
	   status = STR_UPCASE (newname, newname_lc)

* check that the name isnt already in the data set.
           CALL FIND_VAR_NAME ( dset, newname, idum, testvar )
	   len_rbuff = TM_LENSTR1(newname)
           IF (  testvar .NE. munknown_var_name ) GOTO 5400

	   IF ( .NOT.TM_LEGAL_NAME(newname) ) THEN
	      risc_buff = vname
	      len_rbuff = LEN(vname)
              GOTO 5200
           ENDIF
	ELSE
	   newname = char_init128
	ENDIF

* SET VAR/SCALEFAC only for NetCDF datasets

	loc = qual_given( slash_set_var_scale )
	IF ( loc .GT. 0 ) THEN

           IF (dset .EQ. unspecified_int4) GOTO 5700
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

           IF (dset .EQ. unspecified_int4) GOTO 5700
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

* SET VAR/TITLE=

	loc = qual_given( slash_set_var_title )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      title, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   title = char_init80
	ENDIF

* SET VAR/UNITS=

	loc = qual_given( slash_set_var_unit )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      units, status )
           IF ( status .NE. ferr_ok ) RETURN
	ELSE
	   units = char_init16
	ENDIF

* SET VAR/GRID=

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

* SET VAR/BAD=

	loc = qual_given( slash_set_var_bad )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
* *kob* 7/03 - Add check for bad=NaN
	   CALL EQUAL_STRING (cmnd_buff(qual_start(loc):qual_end(loc)),
     .                         is_nan, status)
	   IF (TM_HAS_STRING(is_nan, 'NaN')) THEN
              CALL SET_NAN(bad)
	   ELSE
	      CALL EQUAL_VAL( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			   bad, status )
	      IF ( status .NE. ferr_ok ) RETURN
	   ENDIF
	ELSE
	   bad = real4_init
	ENDIF

* SET VAR/OUTTYPE= INTYPE, FLOAT, INT, SHORT

	loc = qual_given( slash_set_var_outtype )
	IF ( loc .GT. 0 ) THEN
	   have_mods = .TRUE.
           CALL EQUAL_STR_LC( cmnd_buff(qual_start(loc):qual_end(loc)),
     .			      flag, status )
           IF ( status .NE. ferr_ok ) RETURN
           i = STR_UPCASE (typflag, flag)

           IF (typflag(1:5) .EQ. 'INPUT'  ) THEN   ! input type
              outtype_flag = 99  ! do not change from incoming type
           ELSE IF (typflag(1:3) .EQ. 'BYT' .OR. typflag(1:4) .EQ. 'INT1') THEN
              outtype_flag = 1   ! NC_BYTE
           ELSE IF (typflag(1:3) .EQ. 'SHO' .OR. typflag(1:4) .EQ. 'INT2') THEN
              outtype_flag = 3   ! NC_SHORT
           ELSE IF (typflag(1:3) .EQ. 'INT' .OR. typflag(1:4) .EQ. 'INT4') THEN
              outtype_flag = 4   ! NC_INT (4-byte integer)
           ELSE IF (typflag(1:3) .EQ. 'FLO' .OR. typflag(1:3) .EQ. 'REA') THEN
              outtype_flag = 5   ! NC_FLOAT
           ELSE IF (typflag(1:3) .EQ. 'DOU') THEN
              CALL WARN 
     .         ('Converting single precision variable to double on output')
              outtype_flag = 6   ! NC_DOUBLE

           ELSE
              GOTO 5900
           ENDIF

* get dataset from varname.attname
* evaluate the context without computing the expression to get the dataset

           IF (dset .EQ. unspecified_int4) THEN

              CALL GET_NEW_CX( cx_last, cx_cmnd, .TRUE., status )
              IF ( status .NE. ferr_ok ) GOTO 5000
 
              CALL EVAL_CONTEXT ( cx_cmnd, varattname, status )
              IF ( status .NE. ferr_ok ) GOTO 5000

              cx = is_cx( 1 )
              dset = cx_data_set(cx) 
              IF (dset .EQ. pdset_irrelevant) THEN
                 dset = cx_data_set(cx_last) 
                 cx = cx_last
              ENDIF

              IF (dset .EQ. unspecified_int4) dset = -1 ! user variables

           ENDIF

* Make sure we have varid for varname, then reset the variable type.

           CALL CD_GET_VAR_ID (dset, vname, varid, status)

* When the user requests double precision output, write an attribute to the
* file explaining that it contains data converted from float.

           IF (typflag(1:3) .EQ. 'DOU')
     .       CALL CD_PUT_NEW_ATTR (dset, varid, 
     .        'Ferret_Precision_Note', 2, 61, 1, 
     .        'This variable written by Ferret was converted from '//
     .        'FLOAT to DOUBLE', 1., status )

           IF (status .NE. ferr_ok) GOTO 5000

* Get input type, store in variable outtype_flag
           IF (outtype_flag .EQ. 99) THEN
              CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, 
     .                  nvdims, vdims, nvatts, coordvar, varoutflag, 
     .                  status) 
              outtype_flag = vartype 
           ENDIF

           CALL CD_SET_VAR_TYPE (dset, varid, outtype_flag, status )
           IF (status .NE. ferr_ok) GOTO 5000

        ENDIF

* Apply the changes in Ferret structures, and in the linked-list structure.

        IF (dset .EQ. unspecified_int4) dset = -1 ! user variable
        CALL CD_GET_VAR_ID (dset, vname, varid, status)

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

              attlen = 1
              attname = 'missing_value'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
              ENDIF
              attname = '_FillValue'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
              ENDIF

	   ENDIF
*	   IF (newname.NE. char_init16 ) ds_var_code ( var ) = newname
	   IF (newname.NE. char_init128 )
     .         CALL string_array_modify( ds_var_code_head, var,
     .                                   newname, LEN(newname))

	   IF ( units .NE. char_init16 ) THEN
              ds_var_units( var ) = units

              attlen = TM_LENSTR1(units)
              attname = 'units'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           units, dummy, status)
              ENDIF
           ENDIF

	   IF ( title .NE. char_init80 ) THEN
              ds_var_title( var ) = title

              attlen = TM_LENSTR1(title)
              attname = 'long_name'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           title, dummy, status)
              ENDIF
           ENDIF

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
	   IF (newname.NE. char_init128 ) 
     .         CALL string_array_modify(uvar_name_code_head,
     .                                  var,
     .                                  newname,
     .                                  LEN(newname))

           dset = -1 ! user variables
           CALL CD_GET_VAR_ID (dset, vname, varid, status)

	   IF ( units .NE. char_init16 ) THEN
              uvar_units( var )  = units

              attlen = TM_LENSTR1(units)
              attname = 'units'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           units, dummy, status)
              ELSE
                 attype = NCCHAR
                 attoutflag = 1
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, 
     .              attlen, attoutflag, units, dummy, status )
              ENDIF
           ENDIF
     
	   IF ( title .NE. char_init128 ) THEN
              uvar_title( var ) = title

              attlen = TM_LENSTR1(title)
              attname = 'long_name'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           title, dummy, status)
              ELSE
                 attype = NCCHAR
                 attoutflag = 1
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, 
     .              attlen, attoutflag, title, dummy, status )
              ENDIF
           ENDIF

	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_ALL_UVARS	 ! delete invalidated uvars
	      uvar_bad_data(var) = bad

              attlen = 1
              attname = 'missing_value'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
              ELSE
                 attype = NCFLOAT
                 attoutflag = 1
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, 
     .              attlen, attoutflag, buf, bad, status )
              ENDIF

              attname = '_FillValue'
              CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
              IF (attid .GT. 0) THEN
                 CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             attname, attype, attlen, attoutflag, status )
                 CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .           buf, bad, status)
              ELSE
                 attype = NCFLOAT
                 attoutflag = 1
                 CALL CD_PUT_NEW_ATTR (dset, varid, attname, attype, 
     .              attlen, attoutflag, buf, bad, status )
              ENDIF
	   ENDIF

	ENDIF

! Replace the name in the linked-list structure.
	IF (newname .NE. char_init128 ) 
     .      CALL CD_RENAME_VAR (dset, varid, newname_lc, status)

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
     
 5900	CALL ERRMSG( ferr_unknown_arg, status,
     .'Can SET VARIABLE/OUTTYPE=INPUT, DOUBLE, FLOAT, INT, SHORT, BYTE',
     .               						*5000 )

	END

