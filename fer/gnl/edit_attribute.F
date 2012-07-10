

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
* V64  10/09 *acm* longer string length for grid_name

	include 'tmap_dims.parm'
*TMAP_DSET.PARM
*
* parameters used in "TMAP-formatted" data sets
*
* revision 0.00 - 05/07/87 - by Mark Verschell for PMEL/TMAP
* revision 0.01 - 11/21/88 - Changed variable initialization to char_initXX
* revision 0.02 - 3/99 - *kob* added char_init64
* revision 0.03 - 4/99 - *kob* added char_init256
* V500 - 7/99 *sh* ==> changed char_init to "%%" so that all of the
*			char_init*N are effectively the same definition
* V581 - 3/05 *acm* add char_init_1024
* V68    1/12 *acm* Real variables should be declared REAL (conversion to double prec.)
*        3/12 *acm* Add E and F dimensions (use nferdims in tmap_dims.parm)
* V680   5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE


* codes for data sets
	INTEGER		set_not_open,set_open,file_not_open
	PARAMETER      (set_not_open = -9)	! shows available set storage
	PARAMETER      (set_open = -19)		! shows available set storage
	PARAMETER      (file_not_open = -7)	! shows data file not open

* codes for end of namelist blocks
	CHARACTER*4	end_of_aux
	CHARACTER*8	end_of_variables
	CHARACTER*40	end_of_stepfiles, end_of_varfiles
	CHARACTER*60	end_of_grids
	PARAMETER	(end_of_variables = '**END OF VARS**')
	PARAMETER	(end_of_aux = '**END OF AUX**')
	PARAMETER	(end_of_stepfiles = '**END OF STEPFILES**')
	PARAMETER	(end_of_varfiles = '**END OF VARFILES**')
	PARAMETER	(end_of_grids = '**END OF GRIDS**')

* codes for common block initialization
	CHARACTER*80	char_init
	CHARACTER*20	sevens
	CHARACTER	char_init01*01, char_init02*02, char_init04*04,
     .			char_init08*08,	char_init12*12, char_init16*16,
     .			char_init20*20,	char_init23*23, char_init24*24,
     .			char_init30*30, char_init40*40, char_init60*60,
     .			char_init64*64, char_init80*80, char_init128*128,
     .                  char_init256*256, char_init512*512, 
     .                  char_init1024*1024
	INTEGER*2	int2_init
	INTEGER*4	int4_init
	REAL		real4_init
	REAL*8		real8_init
	PARAMETER	(sevens = '77777777777777777777')
! changed this definition because use of char_init and char_initN was
! inconsistent in the code. This way makes all the definitions the same
! (except char_init01)
!	PARAMETER	(char_init = sevens//sevens//sevens//sevens)
	PARAMETER	(char_init = '%%')
	PARAMETER	(char_init01 = char_init)
	PARAMETER	(char_init02 = char_init)
	PARAMETER	(char_init04 = char_init)
	PARAMETER	(char_init08 = char_init)
	PARAMETER	(char_init12 = char_init)
	PARAMETER	(char_init16 = char_init)
	PARAMETER	(char_init20 = char_init)
	PARAMETER	(char_init23 = char_init)
	PARAMETER	(char_init24 = char_init)
	PARAMETER	(char_init30 = char_init)
	PARAMETER	(char_init40 = char_init)
	PARAMETER	(char_init60 = char_init)
	PARAMETER	(char_init64 = char_init)
	PARAMETER	(char_init80 = char_init)
	PARAMETER	(char_init128= char_init)
	PARAMETER	(char_init256= char_init)
	PARAMETER	(char_init512= char_init)
	PARAMETER	(char_init1024= char_init)
	PARAMETER	(int2_init = -7777)
	PARAMETER	(int4_init = -7777)
	PARAMETER	(real4_init = -7.77E-07)
	PARAMETER	(real8_init = -7.77E-07)
	
* record length parameters
	INTEGER		default_reclen,max_reclen,max_tapereclen
	PARAMETER      (default_reclen = 162,	! for GFDL Phil/Seig output
     .			max_reclen = 4095,	! VMS restriction
     .			max_tapereclen = 2499)	! VMS restriction

* bad/missing data indicators
	REAL		bad_r4
	REAL*8		bad_r8
	PARAMETER	(bad_r4 = -1.E34 , bad_r8 = -1.E34)

* zero representations
	INTEGER*1		i1zero
	INTEGER		i4zero
	PARAMETER	(i1zero = 0, i4zero = 0)

* file format codes
	INTEGER		GFDL_3D,pacific_grids,model_output
	CHARACTER*2	grids_at_times,time_series
	CHARACTER*4	pds_type_gt,pds_type_ts,pds_type_ez
	INTEGER*1		regular,irregular
	PARAMETER	(grids_at_times = 'GT',
     .			 time_series = 'TS',
     .			 model_output = 1,
     .			 GFDL_3D = 1,
     .			 pacific_grids = 1,
     .			 regular = 1,
     .			 irregular = 2,
     .			 pds_type_gt = '  GT',
     .			 pds_type_ts = '  TS',
     .			 pds_type_ez = '  EZ' )

* codes for heading records
	INTEGER		plogo,ptitl,pform,pback,pmess,psumm,pvari,
     .			pgrid,pcomm,pindx,pxtra,nbefore_vari
	PARAMETER      (plogo = 1,
     .			ptitl = 2,
     .			pform = 3,
     .			pback = 4,
     .			pmess = 5,
     .			psumm = 6,
     .			pvari = 7,
     .			pcomm = 8,
     .			pgrid = 9,
     .			pindx = 10,
     .			pxtra = 11)
	PARAMETER	(nbefore_vari = 6)	! records preceeding VARI recs

* codes for variables
	INTEGER		ptemp,psalt,pu,pv,pw,ptaux,ptauy,ppsi,pairt,
     .			ptstar_s, ptstar_n, psstar_s, psstar_n
	PARAMETER      (ptemp = 1,
     .			psalt = 2,
     .			pu    = 3,
     .			pv    = 4,
     .			pw    = 5,
     .			ptaux = 6,
     .			ptauy = 7,
     .			ppsi  = 8,
     .			pairt = 9,
     .			ptstar_s = 10,
     .			ptstar_n = 11,
     .			psstar_s = 12,
     .			psstar_n = 13 )

* codes for standard grids and "No defining grid error"
	INTEGER		pps1dxt,pps1dyt,pps1dzb,pps3dtemp,pps3duv,
     .			pps3dw,	pps1dxv,pps1dyv,pps1dz, pps2dtemp,
     .			pps2duv
	INTEGER		no_grid_err
	PARAMETER      (pps1dxt   = 1,
     .			pps1dyt   = 2,
     .			pps1dzb   = 3,
     .			pps1dxv   = 4,
     .			pps1dyv   = 5,
     .			pps1dz    = 6,
     .			pps3dtemp = 7,
     .			pps3duv   = 8,
     .			pps3dw    = 9,
     .			pps2dtemp = 10,
     .			pps2duv   = 11)
	PARAMETER      (no_grid_err = -1)

* codes used at time steps
	CHARACTER*4	step_code
	INTEGER		nintro_recs
	PARAMETER	(nintro_recs = 1 , step_code = 'STEP')

* codes used for time<->date conversion
	INTEGER		to_year, to_month, to_day, to_hour, to_minute,
     .			to_second, full_prec
	PARAMETER	(to_year	= 6,
     .			 to_month	= 5,
     .			 to_day		= 4,
     .			 to_hour	= 3,
     .			 to_minute	= 2,
     .			 to_second	= 1,
     .			 full_prec	= 2)

* codes for auxiliary data sets
	INTEGER		main_set
	PARAMETER	(main_set	= 0)
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret.parm'
	include 'errmsg.parm'
        include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'xrisc.cmn'
	include 'xpyvar_info.cmn'
      INCLUDE 'netcdf.inc'

* local parameter declarations:
      INTEGER  max_line_len
      PARAMETER ( max_line_len = 2048 )


* calling argument declarations:

      INTEGER mr, attype_spec, dset, status
      REAL grid_data(mr_lo_ss(mr,1):mr_hi_ss(mr,1),
     .               mr_lo_ss(mr,2):mr_hi_ss(mr,2),
     .               mr_lo_ss(mr,3):mr_hi_ss(mr,3),
     .               mr_lo_ss(mr,4):mr_hi_ss(mr,4),
     .               mr_lo_ss(mr,5):mr_hi_ss(mr,5),
     .               mr_lo_ss(mr,6):mr_hi_ss(mr,6))
      CHARACTER*(*) varattname

* local variable declarations: 
      
      LOGICAL TM_HAS_STRING, 
     .        valid, do_err, coordvar, have_mods, ez, ncfile
      INTEGER MR_DIM_LEN, TM_LENSTR1, STR_SAME, varid, idim, ndims,
     .        attlen, attype, span, lo(nferdims), hi(nferdims), del(nferdims), slen, vlen, 
     .        attlen_old, attype_old, attoutflag, vartype, nvdims,
     .        nvatts, vdims(8), all_outflag, cat, var, tmap_status,
     .        grid, mod_cx, attid
      REAL dummy, bad, scalefac, offset
      CHARACTER GET_STRING_ELEMENT*2048, TM_FMT*2048, attbuff*2048, 
     .          errbuff*256, varname*512, attname*128, aname*128

      CHARACTER	grid_name*64, title*128, units*64, newname*128

! Check that the expresssion is 1-D and if it is a string, just 1 value.

      ndims = 0
      attlen = 0
      DO 30 idim = 1, nferdims
         span = MR_DIM_LEN(idim, mr )
         valid = mr_lo_ss( mr,idim ) .NE. unspecified_int4
         IF (span .GT. 0 .AND. valid ) ndims = ndims + 1
         attlen = MAX(attlen, span)
 30   CONTINUE

* * * * * DETERMINE INDEX LIMITS * * *
      DO 40 idim = 1, nferdims
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
            attbuff = TM_FMT(
     .                grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)),
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
     .	            lo(5), hi(5), lo(6), hi(6), 
     .              grid_data, lo(1), lo(2), lo(3),
     .              lo(4), lo(5), lo(6), 
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
     .         attbuff, grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)), 
     .	       status)  

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
     .         grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6)), status)  
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
           scalefac = grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
        ENDIF

     	IF ( STR_SAME(attname, 'add_offset') .EQ. 0 ) THEN
           have_mods = .TRUE.
           offset = grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
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
           bad = grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
        ENDIF
     
     	IF ( STR_SAME(attname, '_FillValue') .EQ. 0 ) THEN
           have_mods = .TRUE.
           bad = grid_data(lo(1),lo(2),lo(3),lo(4),lo(5),lo(6))
        ENDIF

* Get var and cat
	CALL PARSE_NAM_DSET( varname, cx_last, dset,
     .			     cat, var, mod_cx, status )

        IF (var .EQ. munknown_var_name)  GOTO 5000
	IF ( cat .EQ. cat_file_var ) ez = ds_type(dset) .EQ. pds_type_ez

	   ncfile = (TM_HAS_STRING(ds_type(dset), 'CDF') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'MC') .OR.
     .               TM_HAS_STRING(ds_type(dset), 'AGG') .OR.
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

        ELSEIF ( cat .EQ. cat_pystat_var ) THEN

           IF ( units .NE. char_init16 ) pyvar_units( var )  = units
           IF ( title .NE. char_init80 ) pyvar_title( var )  = title
           IF ( bad  .NE. real4_init  ) THEN
              CALL PURGE_ALL_UVARS       ! delete (possibly dependent) uvars
              pyvar_missing_flag(var) = bad
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
