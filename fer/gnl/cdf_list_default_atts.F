	SUBROUTINE CDF_LIST_DEFAULT_ATTS (dset, varid, varcode,  
     .          cx, cdfid, outtype, all_outflag, itsa_uvar, 
     .          got_title, output_title, got_history, output_history, 
     .          got_units, output_units, status )
     

* 2/2019  Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group

* On writing to netCDF, write out the attributes that are standard with
* all variables:
*
* - default missing/_FillValues variables -- may be rewritten later 
*    based on what's in the source dataset
* - variable title
* - long_name_mod with subset information
* - variable units
* - variable history, dataset of origin

        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xcontext.cmn'
	include 'xvariables.cmn'

	LOGICAL got_title, output_title, got_history, output_history, 
     .          got_units, output_units, itsa_uvar
	INTEGER dset, varid, outtype, cx, cdfid, all_outflag, status
	CHARACTER*(*) varcode


	LOGICAL NC_GET_ATTRIB_DP, NC_GET_ATTRIB, 
     .          got_it, full_precision, get_ds, need_doc
	INTEGER TM_LENSTR1, STR_SAME, 
     .          var, i, attoutflag, no_fill_flag, no_missing_flag, attid, 
     .          attype, attlen, lbuff, idim, maxlen, slen, dset_num
	REAL bad, vals(100), miss_in, fill_in
	CHARACTER VAR_TITLE*140, VAR_TITLE_MOD*140, VAR_UNITS*64, CX_DIM_STR*48, 
     .          c1*1, attname*1024, buff*1024, buff1*1024, buff2*1024

	full_precision = .TRUE.
	maxlen = 1024

	var = 0
	IF (cx .NE. cx_none) var = cx_variable(cx)

	dset_num = dset
	IF (cx .NE. cx_none) dset_num = cx_data_set(cx)

	status = ferr_ok

* If its a uvar or expression, write missing_value and _FillValue
* attributes with default Ferret values.

	IF (varid .LE. 0) THEN

	   attname = 'missing_value'
	   vals(1) = bad_val4
	   attlen = 1
	   attype = outtype
	   CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .                          attname, vals, attlen, attype, status )
	   attname = '_FillValue'
	   vals(1) = bad_val4
	   attlen = 1
	   attype = outtype
	   CALL CD_WRITE_ATTVAL_DP(cdfid, varcode,
     .                          attname, vals, attlen, attype, status )
	ENDIF

* variable title
* May already have this from Ferret variables from attribute structure, or if there
* is no long_name attribute, get it by means of VAR_TITLE.
	
	IF (.NOT. got_title) THEN
	   buff = ' '
	   IF (cx .NE. cx_none) buff = VAR_TITLE( cx )
	   lbuff = TM_LENSTR1( buff )

	   IF ( buff .NE. ' ' .AND. all_outflag .NE. 0 .AND.
     .                output_title ) THEN
	      CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name',
     .                              buff(:lbuff), .FALSE., status )
	      IF ( status .NE. ferr_ok ) GOTO 5100
	      got_title = .TRUE.
	   ENDIF

	ENDIF

* modifier to variable title
* ..."compressing" transforms and limits not given in definitions
	buff = ' '
	lbuff = 1
	IF (cx .NE. cx_none) THEN
	   DO idim = 1, nferdims
	      need_doc =  cx_trans(idim,cx) .GT. trans_compress_code
	      IF ( itsa_uvar ) need_doc = need_doc
     .                    .OR. uvar_given(idim,var) .GT. uvlim_needed
	      IF ( need_doc ) THEN
	         IF ( lbuff .GT. 1 ) THEN      ! append a comma ?
	            buff = buff(:lbuff)//', '
	            lbuff = MIN( 140, lbuff+2)
	         ENDIF
	         buff = buff(:lbuff)//CX_DIM_STR(idim, cx, ':',full_precision,i)
	         lbuff = TM_LENSTR1( buff )
	      ENDIF
	   ENDDO

* ...other variable modifiers
	   c1 = VAR_TITLE_MOD( cx )
	   IF ( c1 .NE. ' ' ) THEN
	      IF ( lbuff .GT. 1 ) THEN
	         buff = buff(:lbuff)//', '
	         lbuff = MIN( 140, lbuff+2)
	      ENDIF
	      buff = buff(:lbuff)//VAR_TITLE_MOD( cx )
	      lbuff = TM_LENSTR1( buff )
	   ENDIF
	ENDIF

* ... write it to the file
	IF ( lbuff .GE. 2 .AND. all_outflag .NE. 0) THEN

* ..... unless the user has created a long_name_mod attribute, or asked that writing
*       it be turned off. See if long_name_mod attribute has been created.

	   attoutflag = 1
	   slen = TM_LENSTR1(varcode)
	   buff2 = varcode(1:slen)//'.long_name_mod'
	   IF (varid .GT. 0) THEN
	      CALL BREAK_VARATTNAME (buff2, dset, 
     .               buff1, attname, varid, .FALSE., status)

	      IF (status .EQ. ferr_ok) THEN

	         CALL CD_GET_VAR_ATT_ID (dset, varid, 'long_name_mod', attid, status)
	         IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .               attname, attype, attlen, attoutflag, status )

	         IF (attoutflag .EQ. 1) got_it = NC_GET_ATTRIB 
     .                ( dset, varid, attname, .FALSE., varcode, 
     .                  maxlen, attlen, attoutflag, buff(2:), vals)
	         lbuff = TM_LENSTR1( buff )
	      ENDIF
	   ENDIF
	   status = ferr_ok

	   IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .         CALL CD_WRITE_ATTRIB(cdfid, varcode, 'long_name_mod',  
     .                buff(2:lbuff), .FALSE., status )

	   IF ( status .NE. ferr_ok ) GOTO 5100
	ENDIF

* units
	IF (.NOT.got_units .AND. all_outflag.NE.0 .AND.  output_units ) THEN
	   buff = ' '
	   IF (cx .NE. cx_none) buff = VAR_UNITS( cx )
	   lbuff = TM_LENSTR1( buff )

	   IF ( buff .NE. ' ' .AND. all_outflag .NE. 0 .AND. output_units ) THEN
	      CALL CD_WRITE_ATTRIB(cdfid, varcode, 'units',
     .                              buff(:lbuff), .FALSE., status )
	      IF ( status .NE. ferr_ok ) GOTO 5100
	   ENDIF
	ENDIF

* data set of origin
	IF ( dset .NE. pdset_irrelevant .AND. dset.NE.pdset_uvars) THEN
       
* ..... unless the user has created a history attribute, or asked that writing
*       it be turned off. See if there is a history attribute.
*       If the history attribute is just "from dsetname", or is not present,
*       then write the dataset and its path or URL.

	   attoutflag = 1
	   IF (varid .GT. 0) THEN
	      slen = TM_LENSTR1(varcode)
	      buff = varcode(1:slen)//'.history'
	      CALL BREAK_VARATTNAME (buff, dset, buff1, attname,
     .                   varid, .FALSE., status)

	      IF (status .EQ. ferr_ok) THEN
	         CALL CD_GET_VAR_ATT_ID (dset, varid, 'history', attid, status)
	         IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, attid, 
     .                     attname, attype, attlen, attoutflag, status )

	         IF (attoutflag .EQ. 1) got_it = NC_GET_ATTRIB (
     .		  	     dset, varid, attname, .FALSE., varcode,
     .		  	     maxlen, attlen, attoutflag, buff, vals)
	         lbuff = TM_LENSTR1( buff )
	         IF (buff .EQ. ' ') THEN
	            buff = 'From '
	            CALL GET_DSET_NAME( dset, buff(6:), lbuff )
	            IF (buff(6:7) .EQ. './') 
     .                 CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
	            lbuff = TM_LENSTR1( buff )
	         ELSE
	            get_ds = .FALSE.
	            IF (STR_SAME(buff(1:5), 'From ') .EQ. 0) THEN
	               CALL GET_SHORT_DSET_NAME( dset, buff1, lbuff )
	               IF (STR_SAME(buff(6:attlen), buff1) .EQ. 0) get_ds = .TRUE.
	            ENDIF
	         ENDIF
	      ENDIF

	      IF (status .NE. ferr_ok .OR. get_ds) THEN 
	         buff = 'From '
	         CALL GET_DSET_NAME( dset, buff(6:), lbuff )
	         IF (buff(6:7) .EQ. './') 
     .                 CALL GET_SHORT_DSET_NAME( dset, buff(6:), lbuff )
	         lbuff = TM_LENSTR1( buff )
	      ENDIF
	   
	   ELSE
	      buff = 'From '
	      CALL GET_DSET_NAME( dset_num, buff(6:), lbuff)
	      IF (buff(6:7) .EQ. './') CALL GET_SHORT_DSET_NAME
     .                 ( dset_num, buff(6:), lbuff )
	      lbuff = TM_LENSTR1( buff )
	   ENDIF ! varid .gt. 0

	   status = ferr_ok

	   IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0) THEN
	      lbuff = TM_LENSTR1( buff )
	      CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
	   ENDIF
	   IF ( status .NE. ferr_ok ) GOTO 5100

	ELSEIF ( dset_num .NE. pdset_irrelevant .AND.
     .           dset_num .NE. pdset_uvars) THEN

* Write history attribute containing the data set of origin for user variable, 
* if the user has not created a history attribute for the variable.

	   IF (.NOT.got_history .AND. all_outflag.NE.0 .AND. 
     .         output_history ) THEN

	      attoutflag = 1
	      buff = 'From '
	      CALL GET_DSET_NAME(dset_num, buff(6:), lbuff)
	      IF (buff(6:7) .EQ. './') CALL GET_SHORT_DSET_NAME
     .                 ( dset_num, buff(6:), lbuff )
	      lbuff = TM_LENSTR1( buff )
	      status = ferr_ok

	      IF (all_outflag .NE. 0 .AND. attoutflag .NE. 0)
     .                 CALL CD_WRITE_ATTRIB(cdfid, varcode, 'history', 
     .                              buff(:lbuff), .FALSE., status )
	      IF ( status .NE. ferr_ok ) GOTO 5100
	   ENDIF
	ENDIF


 5100	RETURN	
	END
