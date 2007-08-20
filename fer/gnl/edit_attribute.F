      SUBROUTINE EDIT_ATTRIBUTE( grid_data, mr, varattname, dset, 
     .         attype_spec, status )

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
*  Change an attribute for a variable: can change value(s), and/or type

* programmer - Ansley Manke
* NOAA/PMEL, Seattle, WA - TMAP
* 9/2005
* 3/2006 Allow commands equivalent to SET AXIS/modulo/depth/units etc.
* V600  4/06 *acm* - additional declarations found by Solaris compiler

	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
        include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
      INCLUDE 'netcdf.inc'

* local parameter declarations:
      INTEGER  max_line_len
      PARAMETER ( max_line_len = 2048 )


* calling argument declarations:

      INTEGER mr, attype_spec, dset, status
      REAL grid_data(mr_lo_ss(mr,1):mr_hi_ss(mr,1),
     .               mr_lo_ss(mr,2):mr_hi_ss(mr,2),
     .               mr_lo_ss(mr,3):mr_hi_ss(mr,3),
     .               mr_lo_ss(mr,4):mr_hi_ss(mr,4))
      CHARACTER*(*) varattname

* local variable declarations: 
      
      LOGICAL TM_HAS_STRING, 
     .        valid, do_err, coordvar, have_mods, ez, ncfile
      INTEGER MR_DIM_LEN, TM_LENSTR1, STR_SAME, varid, idim, ndims,
     .        attlen, attype, span, lo(4), hi(4), del(4), slen, vlen, 
     .        attlen_old, attype_old, attoutflag, vartype, nvdims,
     .        nvatts, vdims(8), all_outflag, cat, var, tmap_status,
     .        grid, mod_cx, attid
      REAL dummy, bad, scalefac, offset
      CHARACTER GET_STRING_ELEMENT*2048, TM_FMT*2048, attbuff*2048, 
     .          errbuff*256, varname*512, attname*128, aname*128

      CHARACTER	grid_name*16, title*128, units*64, newname*128

! Check that the expresssion is 1-D and if it is a string, just 1 value.

      ndims = 0
      attlen = 0
      DO 30 idim = 1, 4
         span = MR_DIM_LEN(idim, mr )
         valid = mr_lo_ss( mr,idim ) .NE. unspecified_int4
         IF (span .GT. 0 .AND. valid ) ndims = ndims + 1
         attlen = MAX(attlen, span)
 30   CONTINUE

* * * * * DETERMINE INDEX LIMITS * * *
      DO 40 idim = 1, 4
         lo(idim) = mr_lo_ss( mr, idim )
         hi(idim) = mr_hi_ss( mr, idim )
         del(idim)= 1
 40   CONTINUE

      IF (ndims .LT. 0 .OR. ndims .GT. 1) GOTO 5100

* Reconcile data types for the attribute. If requested a float attribute
* but the expression is a string, it is an error.  If a string was requested,
* we can convert a numeric value to string below.

      IF (attype_spec .EQ. ptype_float  .AND. 
     .    mr_type(mr) .NE. ptype_float) GOTO 5300

      attype = NCFLOAT
      IF (mr_type(mr) .EQ. ptype_string .OR.
     .    attype_spec .EQ. ptype_string ) attype = NCCHAR

* See if the attribute already exists for the variable.

      do_err = .TRUE.
      CALL BREAK_VARATTNAME ( varattname, dset, varname, 
     .              attname, varid, do_err, status )
      IF (status .NE. ferr_ok) GOTO 5400

      CALL CD_GET_VAR_INFO (dset, varid, varname, vartype, nvdims, 
     .            vdims, nvatts, coordvar, all_outflag, status)

      IF (coordvar) THEN

* Do not allow user to change the direction of an axis.

        IF (STR_SAME(attname, 'AXIS')           .EQ. 0  .OR.
     .      STR_SAME(attname, 'CARTESIAN_AXIS') .EQ. 0) THEN
     
            vlen = TM_LENSTR1(varname)
            slen = TM_LENSTR1(attname)
            GOTO 5600
        ENDIF

      ENDIF

!  If attribute already exists, change it.

      CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
      IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .             aname, attype_old, attlen_old, attoutflag, status )

      errbuff = varattname
      IF (attype .EQ. NCCHAR ) THEN

         IF (mr_type(mr) .NE. ptype_string) THEN
            attbuff = TM_FMT(grid_data(lo(1), lo(2), lo(3), lo(4)),
     .                8, 40, slen)
            IF (attlen .GT. 1) THEN 
               CALL WARN ('Requested string attribute using a '//
     .                    'numeric expression. Using first value and '//
     .                    'converting to string')
               attlen = 1
            ENDIF

         ELSE
            IF (attlen .GT. 1) GOTO 5200
            attbuff = GET_STRING_ELEMENT(
     .              lo(1), hi(1), lo(2), hi(2),
     .	            lo(3), hi(3), lo(4), hi(4), 
     .              grid_data, lo(1), lo(2), lo(3), lo(4), 
     .              max_line_len, slen)

            dummy = 1.
            attlen = TM_LENSTR1(attbuff)    
         ENDIF

         vlen = TM_LENSTR1(errbuff)
         IF (attype .NE. attype_old) 
     .         CALL WARN ('Changing the type of attribute '//
     .         errbuff(:vlen))
         CALL WARN ('Changing the value of attribute '//
     .         errbuff(:vlen))

         CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .         attbuff, dummy, status)

      ELSE 
         attype = NCFLOAT
         vlen = TM_LENSTR1(errbuff)
         IF (attype .NE. attype_old) 
     .         CALL WARN ('Changing the type of attribute '//
     .         errbuff(:vlen))
         CALL WARN ('Changing the value of attribute '//
     .         errbuff(:vlen))

         IF (attlen .NE. attlen_old .AND. attype .EQ. attype_old) 
     .         CALL WARN ('Changing the length  of attribute '//
     .         errbuff(:vlen))

         CALL CD_REPLACE_ATTR (dset, varid, attname, attype, attlen, 
     .         attbuff, grid_data(lo(1),lo(2),lo(3),lo(4)), status)  

      ENDIF

* For coordinate variables, change attributes as in SET AXIS command.

      IF (coordvar) THEN
         IF (STR_SAME(attname, 'modulo')      .EQ. 0  .OR.
     .       STR_SAME(attname, 'depth')       .EQ. 0  .OR.
     .       STR_SAME(attname, 'positive')    .EQ. 0  .OR.
     .       STR_SAME(attname, 'calendar')    .EQ. 0  .OR.
     .       STR_SAME(attname, 'time_origin') .EQ. 0  .OR.
     .       STR_SAME(attname, 'units')       .EQ. 0) THEN

! input varname may differ from name in linked-list structure by a digit (see isit_coord_var)
           CALL GET_INPUT_VARNAME (varattname, varname)  
           CALL RESET_AXIS (dset, varid, varname, attname, attbuff,
     .            grid_data(lo(1),lo(2),lo(3),lo(4)), status)  
        ENDIF
      ENDIF

* Replace information in Ferret variables as in SET VAR

	grid = unspecified_int4
	bad = real4_init
	scalefac = real4_init
	offset = real4_init
        have_mods = .FALSE.
        ncfile = .FALSE.
        ez = .FALSE.
	newname = char_init128
	title = char_init128
	units = char_init64
	grid_name = char_init16

     	IF ( STR_SAME(attname, 'scale_factor') .EQ. 0 ) THEN
           have_mods = .TRUE.
           scalefac = grid_data(lo(1),lo(2),lo(3),lo(4))
        ENDIF

     	IF ( STR_SAME(attname, 'add_offset') .EQ. 0 ) THEN
           have_mods = .TRUE.
           offset = grid_data(lo(1),lo(2),lo(3),lo(4))
        ENDIF

     	IF ( STR_SAME(attname, 'long_name') .EQ. 0 ) THEN
           have_mods = .TRUE.
           title = attbuff
        ENDIF

     	IF ( STR_SAME(attname, 'units') .EQ. 0 ) THEN
           have_mods = .TRUE.
           units = attbuff
        ENDIF

     	IF ( STR_SAME(attname, 'missing_value') .EQ. 0 ) THEN
           have_mods = .TRUE.
           bad = grid_data(lo(1),lo(2),lo(3),lo(4))
        ENDIF
     
     	IF ( STR_SAME(attname, '_FillValue') .EQ. 0 ) THEN
           have_mods = .TRUE.
           bad = grid_data(lo(1),lo(2),lo(3),lo(4))
        ENDIF

* Get var and cat
	CALL PARSE_NAM_DSET( varname, cx_last, dset,
     .			     cat, var, mod_cx, status )

        IF (var .EQ. munknown_var_name)  GOTO 5000
	IF ( cat .EQ. cat_file_var ) ez = ds_type(dset) .EQ. pds_type_ez

	   ncfile = (TM_HAS_STRING(ds_type(dset), 'CDF') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'MC') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'GT')  )

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

	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
	      ds_missing_flag(var) = ds_bad_flag(var)
	      ds_bad_flag    (var) = bad
	   ENDIF

	   IF ( units .NE. char_init16 ) ds_var_units( var ) = units
	   IF ( title .NE. char_init80 ) ds_var_title( var ) = title

           IF ((scalefac .NE. real4_init) .AND. ncfile) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
              ds_var_scale(var) = scalefac
              ds_var_scaleit(var)       = .TRUE.
           ENDIF
           IF ((offset .NE. real4_init) .AND. ncfile) THEN
	      CALL PURGE_FILE_VAR( var ) ! remove memory-resident variables
	      CALL PURGE_ALL_UVARS	 ! delete (possibly dependent) uvars
              ds_var_off(var)   = offset
              ds_var_scaleit(var)       = .TRUE.
           ENDIF
	ELSE
* ... user-defined var

	   IF ( units .NE. char_init16 ) uvar_units( var )  = units
	   IF ( title .NE. char_init80 ) uvar_title( var )  = title
	   IF ( bad  .NE. real4_init  ) THEN
	      CALL PURGE_ALL_UVARS	 ! delete invalidated uvars
	      uvar_bad_data(var) = bad
	   ENDIF

	ENDIF

* successful completion
	status = ferr_ok




 5000 RETURN
 5100 CALL ERRMSG( ferr_invalid_command, status,
     . 'attribute values must be 1 dimensional: '//
     . errbuff(:vlen),*5000 )

 5200 CALL ERRMSG( ferr_invalid_command, status,
     .  'String attribute may contain just one string: '//
     .  errbuff(:vlen),*5000 )

 5300 CALL ERRMSG( ferr_invalid_command, status,
     .  'Attribute type does not match expression: '//
     .  errbuff(:vlen),*5000 )

 5400 errbuff = varattname
      CALL ERRMSG( ferr_invalid_command, status,
     .  'Attribute does not exist: '//errbuff(:vlen),*5000)

 5500 CALL ERRMSG( ferr_invalid_command, status,
     .  'To change "'// attname(:slen)//
     .  '" for coordinate variable, use SET AXIS instead',*5000)

 5600 CALL ERRMSG( ferr_invalid_command, status,
     .  'Cannot change attribute for direction of axis.',*5000)

      END
