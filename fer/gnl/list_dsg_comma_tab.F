	SUBROUTINE LIST_DSG_COMMA_TAB (dset, cx_list, mr_list, nvar,
     .				      heading, nfeatures, nobs )


*
*  This software was developed by the Science Data Integration Group (SDIG) 
*  of the National Oceanographic and AtmosphericAdministration's (NOAA) 
*  Pacific Marine Environmental Lab(PMEL) as part of Ferret/PyFerret
*  5/1/2019
*
* List data from a single DSG feature collection as a tab- or comma-delimited file
* with a header
* 
*  6/2019 *acm* Write strings in id's, dates without quotes (as ERDDAP does)
*  7/2019 *acm* Future: assign precision per variable, but for now use default or user-set prec.
* V751 7/18 *acm* Call single routine for DSG feature-type names
* V76  1/20 *acm* working with Point-type dsg data 

        include 'netcdf.inc'
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
	include 'slash.parm'
	include 'xrisc.cmn'          ! SPLIT_LIST buffer


* calling argument declarations:
	LOGICAL heading
	INTEGER dset, nvar, nfeatures, cx_list(nvar), mr_list(nvar)

* internal parameter declarations:
	INTEGER maxlen
	PARAMETER (maxlen = 20)   ! length of buffer str

* internal variable declarations:
	LOGICAL process_feature(nfeatures), process_obs(nobs), its_dsg, 
     .		its_cmpnd, has_mask, norow
	LOGICAL instance_coord(4), relevant_coord(4),
     .		instance_var(nvar), point_var(nvar), 
     .		instance_vars_only, point_vars_only, 
     .		ftr_label, do_comma_del, do_tab_del
	INTEGER CGRID_SIZE, TM_LENSTR1, DSG_WHATS_IT, GET_OFFSET_C_STRING_LEN,
     .		i, j, ifeature, ivar, iobs, nobs, idim, cx, grid, slen,
     .		orientation, obsdimlen, coord_lm(4), fvar_varid, base, ptr,
     .		bufflen, line, mr,  gxlate, row_size_lm, clen, nblank,
     .		feature_line, prec_digits(nvar), iwd, 
     .		alen, slen1, slen2, vname_len, list_dsg_digits, dtype, 
     .		idlen, ilen, direc(nvar), vstrlen, paren, coord_ord(4), 
     .		icol, ipad, sdim, blen, status
	REAL	value
	CHARACTER VAR_CODE*128, TM_FMT*48, VAR_UNITS*64,
     .		FMT*20, AX_TITLE*32, AX_TITLE_UNITS*64, SANITARY_VAR_CODE*128, 
     .		CD_DSG_FEATURENAME*20,
     .		strbuff*128, line_buff*10000, vcode(nvar)*128,  bad_buff*32,
     .		col_head_format*64, pstr*15, feature_idname*32, id_buff*128,
     .		tab_or_comma*1, cn*8

	CHARACTER*255	VAR_TITLE, VAR_TITLE_MOD, PAREN_SURROUND, vname_buff

	PARAMETER (vstrlen=40)  ! length of vstr
	CHARACTER vstr*40

	
*internal variable declarations:
        INTEGER flen
        PARAMETER  ( flen = 255 )
	INTEGER*1 fhol(flen)	! c-type Hollerith string buffer


	DATA coord_ord /4,1,2,3/
	

* initialize

        norow = qual_given(slash_list_norow) .GT. 0

* ... column headings for tab and comma-delimited output (9/95)
		
        do_tab_del   = list_fmt_type .EQ. plist_tab_del
        do_comma_del = list_fmt_type .EQ. plist_comma_del

	IF ( do_comma_del ) THEN
	   tab_or_comma = ','
	ELSEIF( do_tab_del ) THEN
	   tab_or_comma = CHAR(9)
	ENDIF

	gxlate = dsg_xlate_grid(dset)
	cx = cx_list(1)
	grid = cx_grid(cx)
	its_dsg = .TRUE.
	CALL TM_DSG_FACTS( grid, orientation, obsdimlen, 
     .			   feature_line, its_dsg, its_cmpnd )
	IF (.NOT. its_dsg) GOTO 5200
	row_size_lm = dsg_loaded_lm(dsg_row_size_var(dset))

	feature_idname = 'feature-id'
	IF (orientation.GE.pfeatureType_Trajectory .AND. 
     .	    orientation.LE.pfeatureType_Point) THEN
	   strbuff = CD_DSG_FEATURENAME(orientation)
	   feature_idname = strbuff(:tm_lenstr1(strbuff)) // '-id'
	ENDIF

	pstr = '               '

* determine precision for outputting values

	list_dsg_digits = MAX(16,list_digits + 2)

	i = qual_given( slash_list_precision ) 
	IF ( i .GT. 0 ) THEN
	   CALL EQUAL_VAL( cmnd_buff(qual_start(i):qual_end(i)),
     .                     value, status )
           IF ( status .NE. ferr_ok ) GOTO 5100
	   list_dsg_digits = MAX(1.,MIN(value, 16.))	! 1 to 16
	ENDIF

* are there any observation vars?  Or is this all instance vars?
	point_vars_only    = .TRUE.   ! a hack to list E=@AVE result
	instance_vars_only = .TRUE.   ! a hack to list instance vars only
	ftr_label = .TRUE.

	DO ivar = 1, nvar
	   cx = cx_list(ivar)
	   grid = cx_grid(cx)
	   instance_var(ivar) = DSG_WHATS_IT(grid) .EQ. pdsg_instance
	   point_var(ivar) = CGRID_SIZE(cx) .EQ. 1
	   IF (.NOT.point_var(ivar)   ) point_vars_only    = .FALSE.
	   IF (.NOT.instance_var(ivar)) instance_vars_only = .FALSE.
	ENDDO

********** as in list_multi ***

* get direction of each variable in the dsg file

	DO ivar = 1, nvar
	   
	   cx = cx_list(ivar)
	   mr = mr_list(ivar)
	   direc(ivar) = no_dim

	   DO idim = 1, nferdims
	      IF ( cx_hi_ss(cx,idim) .GT. cx_lo_ss(cx,idim) ) 
     .	          direc(ivar) = idim
	   ENDDO
	
	ENDDO

* The varid of the feature-id variable
 
	line_buff = SANITARY_VAR_CODE(cat_file_var, dsg_feature_var(dset) )
	CALL CD_GET_VAR_ID  (dset, line_buff, fvar_varid, status)

* which dimensions are relevant to show
	DO idim = 1, 4
	   line = grid_line(idim, gxlate)
	   relevant_coord(idim) =  line.NE.mnormal .AND. .NOT.point_vars_only
	   IF (relevant_coord(idim)) THEN
	      ivar = dsg_coord_var(idim,dset)
	      coord_lm(idim) = dsg_loaded_lm(ivar)  ! line memory table indices
	      instance_coord(idim) = line_dim(line) .EQ. nfeatures

	      IF (instance_vars_only .AND. .NOT.instance_coord(idim))
     .		   relevant_coord(idim) = .FALSE.
	   ENDIF
	ENDDO


* create a mask showing which features to list
	IF (point_vars_only) THEN
	   process_feature(1) = .TRUE.    ! E axis reduced to 1 point
	   ftr_label = (nfeatures .EQ. 1) ! if compressed along ftr dir, don't list feature lab
	ELSE
	   CALL MAKE_DSG_FEATURE_MASK(dset, cx, process_feature, nfeatures)
	ENDIF
	IF (norow) ftr_label = .FALSE.

* Header

	IF (heading) THEN

* data set name and title
	   CALL GET_DSET_NAME( dset, line_buff, alen )
	   CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             DATA SET: '//line_buff(:alen), alen+23)

* ... title of data set
	   CALL GET_DSET_TITLE( dset, line_buff, alen )
	   IF ( alen .GT. 1 ) CALL SPLIT_LIST(pttmode_explct, list_lun,
     .		'             '//line_buff(:alen), alen+13)


* ... then variable titles and any special limits or data set for a variable

	   strbuff = VAR_CODE( cat_file_var, dsg_feature_var(dset))
	   alen = TM_LENSTR1( strbuff )
	   strbuff = feature_idname

	   vname_len = 0
	   DO ivar = 1, nvar
	      cx = cx_list(ivar)
	      strbuff = VAR_CODE( cx_category(cx),cx_variable(cx))
	      vname_len = MAX(TM_LENSTR1(strbuff), vname_len)
	   ENDDO

	   vname_buff = VAR_CODE( cat_file_var, dsg_feature_var(dset))
*          Appending space just to get rid of a nonsense compiler warning
	   strbuff = feature_idname(:TM_LENSTR1(feature_idname))//' '
	   clen = TM_LENSTR1( strbuff )
	   risc_buff = ' '



* ID-var
	   icol = 1
	
	   cn = TM_FMT(FLOAT(icol), 5,8,slen)
	   ipad = MIN(2,3-slen)
	   risc_buff = " Column"//pstr(:ipad)//cn(:slen)//": "
	   slen = TM_LENSTR1(risc_buff)

* if /NOROW then not writing the ID variable and coordinates

	   IF (norow) THEN

	      icol = 0		! Will start lines labeling columns with variable names
	   
	   ELSE
     
	      risc_buff(slen+2:) = vname_buff(:alen)//' is '//strbuff(:clen)
	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)

* names of coordinate-variables

	      DO idim = 1, 4
	         sdim = coord_ord(idim)
	         IF ( relevant_coord(sdim) ) THEN
	            icol = icol + 1
	            cn = TM_FMT(FLOAT(icol), 5,8,slen)
	            ipad = MIN(2,3-slen)

	           IF (sdim .EQ. t_dim) THEN
	               col_head_format = AX_TITLE( sdim, gxlate, alen )
	            ELSE
	               col_head_format = AX_TITLE_UNITS( sdim, gxlate, alen )
                    ENDIF

	            risc_buff = " Column"//pstr(:ipad)//cn(:slen)//': '//col_head_format(:alen)

	            CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)
	         ENDIF
	      ENDDO

	   ENDIF ! norow

* Variables with units and bad-flag

2020	   format ("(A", I0, ",' is ', A)")
	   WRITE(FMT,2020) vname_len

	   DO ivar = 1, nvar
	   
	      icol = icol + 1

	      cx = cx_list(ivar)
	      mr = mr_list(ivar)
	      vname_buff = VAR_CODE( cx_category(cx),cx_variable(cx))

	      strbuff   = VAR_TITLE( cx )
	      clen   = TM_LENSTR1( strbuff )
	      strbuff(clen+1:)  = ' '//PAREN_SURROUND(VAR_UNITS(cx), slen1)
	      clen = clen+1+slen1
	      strbuff(clen+1:) = PAREN_SURROUND(VAR_TITLE_MOD(cx),slen2)
	      clen = clen+slen2
	      risc_buff = ' '

	      cn = TM_FMT(FLOAT(icol), 5,8,slen)
	      ipad = MIN(2,3-slen)
	      risc_buff = " Column"//pstr(:ipad)//cn(:slen)//": "
	      IF (norow .AND. nvar.EQ.1) risc_buff = "Variable: "
	      slen = TM_LENSTR1(risc_buff)

	      WRITE ( risc_buff(slen+2:), FMT ) 
     .           vname_buff(:vname_len), strbuff(:clen)
     	      blen = TM_LENSTR1(risc_buff)

* Show the bad-flag, but not for string variables.

              blen = 0
	      IF ( (mr_type(mr_list(ivar)) .NE. ptype_string)) THEN
                bad_buff = '   BAD FLAG : '// 
     .                  TM_FMT(mr_bad_data(mr),list_dsg_digits,8,blen)
              ENDIF

	      IF (blen .GT. 0) THEN
	         slen = TM_LENSTR1(risc_buff)
	         risc_buff(slen+2:) = bad_buff(:blen+14)
	      ENDIF

	      CALL SPLIT_LIST(pttmode_explct, list_lun, risc_buff, 0)

	   ENDDO  ! nvar
 
	ENDIF  ! heading

* header line for columns  

	slen = 0
	IF (.NOT.norow) THEN
	   line_buff = VAR_CODE( cat_file_var, dsg_feature_var(dset))
	   slen = TM_LENSTR1(line_buff)

	   DO idim = 1, 4
	      sdim = coord_ord(idim)
	      IF ( relevant_coord(sdim) ) THEN	      
	         strbuff = AX_TITLE( sdim, gxlate, alen )
	         line_buff(slen+1:) = tab_or_comma// strbuff(:TM_LENSTR1(strbuff))
	         slen = TM_LENSTR1(line_buff)
	      ENDIF
	   ENDDO
	ENDIF  ! .not. norow


	DO ivar = 1, nvar
	   cx = cx_list(ivar)
	   strbuff = VAR_CODE( cx_category(cx),cx_variable(cx))
	   line_buff(slen+1:) = tab_or_comma//strbuff(:TM_LENSTR1(strbuff))
	   IF (ivar.EQ.1 .AND. norow) line_buff(slen+1:) = strbuff(:TM_LENSTR1(strbuff))
	   slen = TM_LENSTR1(line_buff)
	ENDDO
	IF ( (norow .AND. nvar.GT.1) .OR. (.NOT.norow) ) 
     .	     CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff, 0)


*  ***** LIST THE DATA ******
	base = 0    ! obs index at end of preceding feature
* ... loop over the features
	DO ifeature = 1, nfeatures 
	   nobs = dsg_linemem(row_size_lm)%ptr(ifeature)  ! feature length

* ... have the user's constraints removed this feature?
	   IF (.NOT.process_feature(ifeature)) GOTO 500

* ... each line starts with the Feature ID
	   id_buff = pstr

	   ilen = 0

* Read the feature-id value. If it's numeric turn it into a string.
cc* If it's a string enclose in quotes for consistency with string variables (?)

	   IF (ftr_label) THEN
	      CALL CD_READ_ONE ( dset, fvar_varid, ifeature, dtype, value, 
     .                                 strbuff, bufflen, status )
	      IF (dtype .NE. NF_CHAR) THEN
	         strbuff = TM_FMT(value, 8, 12, bufflen)
cc	      ELSE
cc	         strbuff = '"'//strbuff(:bufflen)//'"'
cc		 bufflen = bufflen+2
	      ENDIF	 

	      id_buff = strbuff(:bufflen)//tab_or_comma
	      ilen = bufflen + 1

* if compressed along the e dimension don't include the feature-id value.

	   ELSE
	      strbuff = '  '
	      bufflen = 1
	      ilen = bufflen
	   ENDIF

* ... get observation-level mask for this feature
	   CALL MAKE_DSG_OBS_MASK(dset, cx, ifeature, base,
     .				  process_obs, nobs)

* ... loop over the observations within each feature
	   DO iobs = 1, nobs
	      ptr = base + iobs  ! pointer (index) into the ragged contig array

* ... have the user's constraints removed this observation?
	      IF (.NOT.process_obs(iobs)) CYCLE

*             Appending space just to get rid of a nonsense compiler warning
	      line_buff = id_buff(:ilen)//' '
	      slen = ilen

* ... insert the relevant formatted coordinates, time first.
cc*     Put the time string in quotes for consistency

	      IF (.NOT. norow) THEN
	         DO idim = 1, 4
	            sdim = coord_ord(idim)
		    IF ( relevant_coord(sdim) ) THEN
	              IF (instance_coord(sdim)) THEN
* ... feature coordinate
	                  value = dsg_linemem(coord_lm(sdim))%ptr(ifeature)
	               ELSE
* ... obs coordinate
	                  value = dsg_linemem(coord_lm(sdim))%ptr(ptr)
	               ENDIF
		       IF (ABS(value) .GE. ABS(bad_val4)) THEN
		          strbuff = '*'
		       ELSE
	                 CALL TRANSLATE_TO_WORLD( value, sdim, gxlate, 
     .					     ax_dec_pt(sdim), strbuff)
	               ENDIF
		       clen = TM_LENSTR1(strbuff)
		       paren = INDEX((strbuff), "(")
		       IF (paren .GT. 0) clen = MIN(clen,paren-1)
cc		       IF (sdim .EQ. t_dim) THEN
cc		          strbuff = '"'//strbuff(:clen)//'"'
cc		          clen = clen + 2
cc	               ENDIF
		       line_buff(slen+1:) = strbuff(:clen)//tab_or_comma
	               slen = slen + clen + 1
	            ENDIF
	         ENDDO

	      ENDIF ! .not. norow

* ... insert the formatted data

	      bufflen = slen
	      DO ivar = 1, nvar
	         mr = mr_list(ivar)
		 
	         IF ( mr_type(mr) .EQ. ptype_string ) THEN
                    vstr = '                '
                    vstr(1:1) = '"'

	            IF (point_var(ivar)) THEN
* ... compressed feature result
	               iwd = 0
	           ELSEIF (instance_var(ivar)) THEN
* ... feature value
	               iwd = ifeature - 1
	            ELSE
* ... obs value
	               iwd = ptr - 1
	            ENDIF

	            slen = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr), iwd )
	            slen = MIN(slen, vstrlen-2)  ! fixed length of vstr...
		    CALL GET_OFFSET_C_STRING(mr_c_pointer(mr),
     .				   iwd,
     .				   vstr(2:),
     .				   slen    )
                    vstr(slen+2:slen+2) = '"'    


	         ELSE

	           IF (point_var(ivar)) THEN
* ... compressed feature result
	               value = memry(mr)%ptr(1)
	            ELSEIF (instance_var(ivar)) THEN
* ... feature value
	               value = memry(mr)%ptr(ifeature)
	            ELSE
* ... obs value
	               value = memry(mr)%ptr(ptr)
	            ENDIF
		    
		    IF (value .EQ. mr_bad_data(mr)) THEN
		       vstr = TM_FMT(mr_bad_data(mr),list_dsg_digits,8,blen)
		    ELSE
		       vstr = TM_FMT(value,list_dsg_digits,list_dsg_digits,slen)
		    ENDIF
		 ENDIF

* remove leading blanks   (better way to handle this?)
		 DO i = 1, vstrlen
                    IF ( vstr( i:i ) .NE. ' ' ) ExIT
		 ENDDO
		 CALL TM_PACK_STRING(vstr, vstr, 1, i, clen)
	         clen = TM_LENSTR1(vstr)

	         line_buff(bufflen+1:) = vstr
	         bufflen = bufflen + clen
		 IF (ivar .LT. nvar) THEN
		    line_buff(bufflen+1:) = tab_or_comma
		    bufflen = bufflen+1
		 ENDIF
	      ENDDO

* ... write out the line
	      CALL SPLIT_LIST(pttmode_explct, list_lun, line_buff, 0)

* ... write only a single line from each feature if there are no ragged arrays
	      IF (instance_vars_only) EXIT	! list just one "obs" line
	   ENDDO

 500	   base = base + nobs
	   IF (point_vars_only) EXIT		! list just one "feature" line
	ENDDO   ! loop over features

 5000	RETURN
 5100	CALL ERRMSG( ferr_invalid_command, status,
     .		'Error getting value from LIST/PRECISION= ',
     .		*5000 )
 5200	CALL ERRMSG( ferr_grid_definition, status,
     .		'Data grid is not a DSG grid ',
     .		*5000 )
	END

