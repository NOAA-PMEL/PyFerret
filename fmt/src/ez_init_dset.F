	SUBROUTINE EZ_INIT_DSET (filename,ezform,variables,types,
     .				 delims,skip,columns,dset,status)

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
*
* Subroutine takes arguments from call and turns it into an entry in the dataset
* and stepfile subroutines. This will allow access to ASCII and unformatted non-
* TMAP data files. This sets up a bare bones entry. All other initialization
* occurs with EZ_MOD_DSET and EZ_READ.
*
* revision 0.00 - 08/03/89
* revision 1.00 - 08/11/89 - Removed calculations involving items that are not
*			     included on SET DATA command to EZ_READ command.
* revision 1.10 - 10/27/89 - *sh* added columns
* revision 1.11 - 01/25/90 - *sh* changed the default format to "FREE"
* Unix/RISC port 3/11/91 *sh*: filename parsing changed (and longer names)
* note: also changed VMS version to save filename extension as well as name
*       (to differentiate snoopy.lis from snoopy.dat)
* rev 1.2 *sh* 7/92 - check for valid, non-duplicated variable names
* rev 1.3 *sh* 11/92 - allow 24 character variable names
* 3/12/93(4?) - lifted restriction on max number of columns
* 12/94 - lifted restriction that # columns must be multiple of # variables
*	  to accomodate /FORMAT=STREAM
* *sh* 3/2/95 - informative messages issued through TM_NOTE (for GUI)
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V530 10/00 *sh* - added delimited reads
* V540 10/01 *sh* - auto-analysis of delimited files to tag MISSING fields
*		    as FLOAT -- makes the file more intelligible to the user
*		  - auto-analysis sets units and titles
*	          - bug fix: if vars_gvn that determines analyzed num_types
*      *acm* 10/01 increase length of ds_name
*      *acm* 10/01  remove VMS include statements
* 1/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an character constant instead
* V552 5/03 *acm*: increase size of var_name to 128 chars
* V602  2/07 *acm* Fix bug 1492, changing attributes of coordinate variables 
* v604 *acm* 7/07 increase length of ds_name, ds_des_name
* V608 *acm* 10/07 Fixes for memory leak; when initializing dataset this
*                  routine was putting vars from other datasets into the
*                  linked-list structure.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 2/10- all warnings and notes to std error not std out

* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V694 1/15 *acm* variable coordvar should be a logical not an integer
* V701 7/16 *acm* ticket 2450. add date-time and euro-date-time field types
* V701  7/16 *kms* Compare ds_name to char_init2048 instead of char_init1024
* V701  7/16 *acm* Ticket 2454. If the user gives lower-case names when opening
*                  an EZ dataset, keep those names on output, etc.
* V72   6/17 *acm* Ticket 2540, inconsistencies in skipping over variables defined
*                  in a Delimited-file read as name and type "-".
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

	CHARACTER*4	ez_rev
	CHARACTER*1	tab
	PARAMETER 	(ez_rev = '1.11')
	PARAMETER     ( tab = CHAR(9))

*
* Arguments:
*	filename	- name of data file
*	variables	- names of variables (and "-" for skipped variables)
*	columns		- number of columns to read (or int4_init)
*	dset		- (output) data set number
*	status		- (output)
*
	CHARACTER*(*)	filename, ezform, variables, types, delims
	INTEGER*4	skip, columns, dset, status
*
* Include files
	include 'xio.cmn_text'
        external xio_data
	include 'ez_lib.parm'
	include 'tmap_dims.parm'
	include 'tmap_dtypes.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'
	include 'xstep_files.cmn_text'
        external xstep_files_data
        include 'xez_info.cmn_text'
	external xez_info_data
        include 'netcdf.inc'

*
* Local definitions
        LOGICAL         TM_LEGAL_NAME, vars_gvn, its_delimited, coordvar
	CHARACTER*24	type_names(maxezvars)
	CHARACTER*128	var_names(maxezvars)
	INTEGER*4	TM_LENSTR1, STR_SAME, STR_CASE_BLIND_COMPARE,
     .			set_cnt, bpos, epos, ncnt, var_pos, num_names,
     .			nvar, jcnt, stp_cnt, nstp, field_type(maxezvars),
     .			num_types, i, j, name_limit, idummy(maxezvars),
     .			l1, l2, ivar, vartype, varid, tlen
	CHARACTER	TM_STRING*13, TM_LEFINT*4, buff*48, 
     .			line_buff*2048

*
* Assign data set number, exiting if maximum number of data sets are open
	DO 100 set_cnt = 1,maxdsets
	  IF (ds_name(set_cnt) .EQ. char_init2048) THEN
	    dset = set_cnt
	    ds_des_name(dset) = filename
* Parse input line to obtain just filename without directory or extension
            epos = LEN(filename)

* Unix: find final pathname slash
            DO 77 bpos = epos, 1, -1
              IF ( filename(bpos:bpos) .EQ. PathSeparator ) GOTO 78
 77         CONTINUE
            bpos = 0    ! no slash
 78         bpos = bpos + 1
	    ds_name(dset) = filename(bpos:epos)
	    GOTO 110
	  ENDIF
  100	CONTINUE
* Too many data sets are open
	GOTO 9000
*
* Load Format record
  110   ds_type(dset)		= '  EZ'
	ds_format(dset)		= '  1A'
	ds_source_class(dset)	= ' '
	ds_source(dset)		= ' '
	ds_subsource(dset)	= ' '
	ds_prog_rev(dset)	= ez_rev
*
* Load Background record
	WRITE (ds_expnum(dset),'(A2,I2.2)') 'EZ',dset
	ds_modnum(dset)		= ' '
	ds_title(dset)		= ' '
	ds_mod_title(dset)	= ' '
	ds_forcing_set(dset)	= ' '
	ds_t0time(dset)		= ' '
	ds_ntegrate_tstep(dset)	= 1
	ds_time_run(dset)	= ' '
	DO 120 ncnt = 1,num_aux
  	   ds_aux_modnum(ncnt,dset)= ' '
  120	CONTINUE
	ds_parm_name(1,dset)	= 'FORMAT'
	ds_parm_text(1,dset)	= 'FREE'
	DO 130 ncnt = 2,num_parm
	  ds_parm_name(ncnt,dset)	= ' '
	  ds_parm_text(ncnt,dset)	= ' '
  130	CONTINUE
*
* time axis
	ds_time_axis(dset) = unspecified_int4
*
* Load message record
	ds_alert_on_open(dset) 	= .FALSE.
	ds_alert_on_output(dset)= .FALSE.
	ds_message(dset)	= ' '

* much needed flag for delimited files
	its_delimited = ezform(1:4) .EQ. 'DELI'
	IF (its_delimited) THEN
	   name_limit = maxezvars
	ELSE
	   name_limit = maxezfreefmtvars
	ENDIF
*
* parse the supplied list of variable names
	num_names = int4_init
	vars_gvn = variables .NE. ' '
	IF (vars_gvn) THEN
	  CALL TM_Parse_string(variables,maxezvars,var_names,num_names)
	  IF (num_names .GT. name_limit) THEN
	    buff = ' '
	    WRITE (buff,1000) num_names
	    CALL WARN( buff ) 
	    buff = ' '
	    WRITE (buff,1002) name_limit
	    CALL WARN( buff ) 
 1000	    FORMAT( 'attempt to initalize',I3,' variables')
 1002	    FORMAT( 'maximum allowed is',I3,' variables')
	    num_names = name_limit
	  ENDIF
*         ... check for illegal and duplicate names (*sh* 7/92)
          DO 510 ncnt = 1,num_names
            IF ( var_names(ncnt) .EQ. '-' ) GOTO 510
            IF ( .NOT.TM_LEGAL_NAME(var_names(ncnt)) ) GOTO 9400
            DO 500 jcnt = 1, ncnt-1
              IF ( var_names(jcnt) .EQ. '-' ) GOTO 500
              IF ( var_names(jcnt) .EQ. var_names(ncnt) ) GOTO 9400
 500        CONTINUE
 510      CONTINUE
	ENDIF

* information needed for delimited file reads: nfields, types
	IF (its_delimited) THEN
	   IF (types  .NE. ' ') THEN
*          ... parse users list of field types
	     CALL TM_Parse_string(types,maxezvars,type_names,num_types)
	     DO 530 i = 1, num_types
	       tlen = MIN(TM_LENSTR1(type_names(i)), 4)
	       DO 520 j = 1, FTYP_NUM_TYPES
	         IF (type_names(i)(1:tlen) .EQ. ftyp_name(j)(1:tlen)) THEN
	            l1 = TM_LENSTR1(type_names(i))
	            l2 = TM_LENSTR1(ftyp_name (j))
	            IF (l1.GT.l2 ) GOTO 9300  ! ilegal type name
	            IF (type_names(i)(:l1) .NE. ftyp_name(j)(:l1)) GOTO 9300
	            field_type(i) = j
	            GOTO 530
	         ENDIF	       
 520	       CONTINUE
*              ... name not recognized
	       GOTO 9300
 530	     CONTINUE
*            ... if only a single type was given use that type for all vars
	     IF (num_types .EQ. 1) THEN
	       IF (vars_gvn) THEN
	         DO 535 i = 2, num_names
 	            field_type(i) = field_type(1)
 535	         CONTINUE
	         num_types = num_names
	       ELSE
*                ... analyze file for just to get number of fields
*	         ... pass null terminated file name string in line_buff
	         CALL TM_FTOC_STRNG( ds_des_name(dset)
     .		     (:TM_LENSTR1(ds_des_name(dset))), line_buff, 2048 )
	         CALL ANAL_FILE(line_buff, line_buff, delims, skip,
     .	                       2000, 2048, num_types, idummy,
     .			       maxezvars)
	         DO i = 2, num_types
	            field_type(i) = field_type(1)
	         ENDDO
	       ENDIF
	     ENDIF
	   ELSE
*            ... analyze file for list of field types
*	     ... pass null terminated file name string in line_buff
	     CALL TM_FTOC_STRNG( ds_des_name(dset)
     .		     (:TM_LENSTR1(ds_des_name(dset))), line_buff, 2048 )
	     CALL ANAL_FILE(line_buff, line_buff, delims, skip,
     .	                       2000, 2048, num_types, field_type,
     .			       maxezvars)
	     IF (vars_gvn) THEN
	       IF (num_names .GT. num_types) THEN
	         GOTO 9330
	       ELSE
	          num_types = num_names  ! use only fields requested by user
	       ENDIF
	     ENDIF
*	     ... reset missing fields to ANAL_MISSING as flag for later treatment
	     DO i = 1, num_types
	       IF (field_type(i) .EQ. FTYP_MISSING)
     .			field_type(i) = FTYP_ANAL_MISSING
	     ENDDO
	   ENDIF
	   IF (vars_gvn) THEN
	      DO 540 i = 1, num_names
 	         IF (var_names(i).EQ.'-') field_type(i) = FTYP_MISSING
 540	      CONTINUE
	      IF (num_names .GT. num_types) THEN
	        DO 545 i = num_types+1, num_names
 		   field_type(i) = FTYP_MISSING
 545	        CONTINUE
	      ENDIF
	   ELSE
*             ... synthesize names for fields
	      num_names = num_types
	      DO 550 i = 1, num_names
	         IF (field_type(i) .EQ. FTYP_MISSING ) THEN
	           var_names(i) = '-'
	         ELSE
	           var_names(i) = 'V'//TM_LEFINT(i,j)
	         ENDIF
 550	      CONTINUE
	   ENDIF
	ELSE
*       ... not a delimited read
	   IF (.NOT.vars_gvn) THEN
*             ... synthesize name for the one variable
	      num_names = 1
	      var_names(1) = 'V1'
	   ENDIF
	   DO 560 i = 1, num_names
 	      field_type(i) = int4_init
 560	   ENDDO
	ENDIF

* Loop through all variable names
	var_pos  = 1
	DO 170 ncnt = 1,num_names
*
* Find the next storage position
	  DO 140 nvar = var_pos,maxvars
	    IF (ds_var_setnum(nvar) .EQ. set_not_open) GOTO 150
  140     CONTINUE
* No room left for new variables
	  GOTO 9100

*
* Load variable record now
  150     var_pos = nvar
          CALL string_array_modify_upcase (ds_var_code_head,
     .                             nvar,
     .                             var_names(ncnt),
     .                             LEN(var_names(ncnt)) )
	  ds_var_title(nvar)	= var_names(ncnt)
	  ds_var_titl_mod(nvar)	= ' '
	  ds_var_units(nvar)	= ' '
	  ds_missing_flag(nvar)	= -1.E34
	  ds_bad_flag(nvar)	= -1.E34
	  DO 160 jcnt = 1,nferdims
  	     ds_ordering(jcnt,nvar)     = jcnt
  160	  CONTINUE
	  ds_precision(nvar)	= 'SINGLE'
	  ds_aux_set_num(nvar)	= 0
* Other calculated or derived data pertaining to variables
	  ds_ndataperrec(nvar)	= int4_init
	  ds_nrecpervar(nvar)	= int4_init
	  ds_nrecb4var(nvar)	= int4_init
	  ds_var_setnum(nvar)	= dset
	  ds_npospervar(nvar)	= 0
* Delimited read field information
	  IF (field_type(ncnt) .EQ. FTYP_CHARACTER) THEN
	    ds_var_type(nvar) = ptype_string
	  ELSE
	    ds_var_type(nvar) = ptype_float
	  ENDIF
	  IF (its_delimited) THEN
	    IF (field_type(ncnt) .EQ. FTYP_ANAL_MISSING) THEN
	       ds_var_titl_mod(nvar) = 'all values missing'
	       field_type(ncnt) = FTYP_NUMERIC
	    ELSEIF (field_type(ncnt) .EQ. FTYP_EURODATE
     .	       .OR. field_type(ncnt) .EQ. FTYP_DATE
     .	       .OR. field_type(ncnt) .EQ. FTYP_DATIME
     .	       .OR. field_type(ncnt) .EQ. FTYP_EDATIME) THEN
	       ds_var_titl_mod(nvar) = 'Julian days since 1-Jan-1900'
	       ds_var_units(nvar) = 'days'
	    ELSEIF (field_type(ncnt) .EQ. FTYP_TIME) THEN
	       ds_var_titl_mod(nvar) = 'Time of day'
	       ds_var_units(nvar) = 'hours'
	    ELSEIF (field_type(ncnt) .EQ. FTYP_LAT) THEN
	       ds_var_titl_mod(nvar) = 'Latitude'
	       ds_var_units(nvar) = 'degrees_north'
	    ELSEIF (field_type(ncnt) .EQ. FTYP_LON) THEN
	       ds_var_titl_mod(nvar) = 'Longitude'
	       ds_var_units(nvar) = 'degrees_east'
	    ENDIF
	  ENDIF

  170	CONTINUE
*
* Save number of variable names (including "-") in data set
	ds_basic_axes(1,dset)	= num_names

* other available storage for later data set mods ...
	ds_basic_axes(2,dset)	= int4_init
	ds_basic_axes(3,dset)	= int4_init
	ds_basic_axes(4,dset)	= int4_init
	ds_basic_axes(5,dset)	= int4_init
	ds_basic_axes(6,dset)	= int4_init

* store number of columns to read
	IF ( columns .EQ. int4_init ) THEN
	   ds_ntuple(dset) 	= num_names
	ELSE
!	   IF ( MOD(columns,num_names).NE.0 ) GOTO 9300    ! removed 12/94
	   ds_ntuple(dset) 	= columns
	ENDIF
*
* Find the next storage position for stepfiles in common
	DO 180 stp_cnt = 1,maxstepfiles
	  IF (sf_setnum(stp_cnt) .EQ. set_not_open) THEN
	    nstp = stp_cnt
	    GOTO 190
	  ENDIF
  180	CONTINUE
* No room left for new stepfiles
	GOTO 9200
*
* Load stepfile record
  190	sf_name(nstp)		= filename
	sf_setnum(nstp)		= dset
	sf_aux_set_num(nstp)	= 0
	sf_nrecperstep(nstp)	= 1
	sf_skip(nstp)		= 0
	sf_regvars(nstp)	= .TRUE.

*
* save special info for delimited read files - struct ptr into ds_file_info_ptr
	IF (its_delimited) THEN
	  CALL SAVE_DELIMITED_INFO( num_names, field_type, delims,
     .				    ds_file_info_ptr(dset)   )
	ENDIF


* Put the information into the linked-list strucure for the dataset
* and its variables. Use the original non-upcased version of the variable name.

        CALL CD_STORE_DSET (dset, status)
        coordvar = .FALSE.
        varid = 0

        DO 200 ivar = 1, maxvars
           IF (ds_var_setnum(ivar) .EQ. set_not_open) GOTO 210
           IF (ds_var_setnum(ivar) .EQ. dset .AND.
     .         STR_SAME(ds_var_code(ivar) , '-' ) .NE. 0) THEN

              varid = varid + 1
	      IF (ds_var_type(ivar) .EQ. ptype_float) THEN
	         vartype = NF_DOUBLE
	      ELSEIF  (ds_var_type(ivar) .EQ. ptype_string) THEN
	         vartype = NF_CHAR
	      ELSE
	         vartype = NF_DOUBLE
              ENDIF

	      line_buff = ds_var_code(ivar)
	      DO ncnt = 1, num_names
	         IF (STR_CASE_BLIND_COMPARE(ds_var_code(ivar), var_names(ncnt)) .EQ. 0) THEN
	            line_buff = var_names(ncnt)
		    EXIT ! from loop
		 ENDIF
	      ENDDO

	      CALL CD_ADD_VAR( dset, line_buff, varid, 
     .          ds_var_title(ivar), ds_var_units(ivar), vartype,
     .          coordvar, ds_bad_flag(ivar), status)
           ENDIF
  200   CONTINUE

* All done
  210   status = merr_ok
	GOTO 9999
*
* ERROR MESSAGES
 9000	CALL TM_ERRMSG ( merr_dsetlim, status, 'EZ_INIT_DSET',
     .			 no_descfile, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxdsets)),
     .			 no_errstring, *9950)

 9100	CALL TM_ERRMSG ( merr_varlim, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *9950)

 9200	CALL TM_ERRMSG ( merr_filim, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxstepfiles)),
     .			 no_errstring, *9950)

 9300	CALL TM_ERRMSG ( merr_syntax, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'Unrecognized field type: '//type_names(i),
     .			 no_errstring, *9310)
 9310	line_buff = ftyp_name(1)
	jcnt = 1   ! string length of list of field types
	DO 9320 i = 2, FTYP_NUM_TYPES
	   line_buff(jcnt+1:) = ','//ftyp_name(i)
	   jcnt = jcnt + 1 + TM_LENSTR1(ftyp_name(i))
 9320	CONTINUE
	CALL WARN('Valid types are: '//line_buff(:jcnt))
	GOTO 9950

 9330	WRITE (line_buff,3100) num_names, num_types
 3100	FORMAT(I4,' var names but only',I3,' fields found in file scan')
	i = TM_LENSTR1(line_buff)
	CALL TM_ERRMSG ( merr_varlim, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 line_buff(:i),
     .			 no_errstring, *9950)

 9400	CALL TM_ERRMSG ( merr_syntax, status, 'EZ_INIT_DSET',
     .			 dset, no_stepfile,
     .			 'illegal or duplicate variable name '
     .                   //var_names(ncnt),
     .			 no_errstring, *9950)

 9950	dset = unspecified_int4

* get out
 9999	RETURN
	END
