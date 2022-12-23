	SUBROUTINE CDF_LIST_DSETVAR_ATTS (dset, itsa_uvar, its_cdf, varid, 
     .          varcode, bad, outtype, cdfid, do_warn, scalefac, addoff,
     .          got_title, output_title, got_history, output_history, 
     .          got_units, output_units, status )

* 2/2019  Ansley Manke
* NOAA/PMEL, Seattle, WA - Science Data Integration Group

* On writing to netCDF, write out the attributes from the source dataset
* as requested by the user, who may set the variable to write all attributes,
* or only specific ones.

* v751 *acm*  6/19 Allow for longer strings as attribute values
C 12/2022 *acm* Code cleanup: Declare variables miss_in and fill_in as arrays 
*               to avoid rank-mismatch errors when compiling under gfortran v10.


        include 'netcdf.inc'
        include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xprog_state.cmn'

	LOGICAL itsa_uvar, its_cdf, do_warn, got_title, output_title, 
     .          got_history, output_history, got_units, output_units
	INTEGER dset, varid, outtype, cdfid, status
	CHARACTER*(*) varcode
	REAL	bad


	LOGICAL NC_GET_ATTRIB_DP, NC_GET_ATTRIB, MATCH_NAME,
     .          got_it, got_fill, got_missing, coordvar
	INTEGER TM_LENSTR1, TM_CHECK_NAN, 
     .          attoutflag, no_fill_flag, no_missing_flag, attid, attype, 
     .          attlen, iatt, nvatts, slen, alen, all_outflag, maxlen, 
     .          lbuff, nvdims, vdims(8), vtype
	REAL vals(100), miss_in(1), fill_in(1), scalefac, addoff
	CHARACTER attname*128, vbuff*128
	CHARACTER*1024 buff

* For error messages
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/


* Set up to write all the attributes from a file ivariable

* Initialize

	maxlen = 1024
	vbuff = varcode
	
	status = NF_INQ_VARID(dset, vbuff, varid)
	CALL CD_GET_VAR_ID (dset, vbuff, varid, status)


* Output the bad flag used by Ferret (and _FillValue which has been made  
* the same as missing_flag)

	vals(1) = bad
	attlen = 1
	got_fill = .FALSE.

* Get attype for missing flag

	attoutflag = 0
	no_missing_flag = 1
	no_fill_flag = 0  ! Write _fillValue unless specifically told not to

	attname = 'missing_value'
	CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	IF (attid .GT. 0) THEN
	   no_missing_flag = 0
	   CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, attname, attype, attlen, attoutflag, status )
	   IF (attoutflag .EQ. 0) no_missing_flag = 1
	   IF (attoutflag .NE. 0) got_it = NC_GET_ATTRIB_DP 
     .                (dset, varid, attname, do_warn, vbuff, attlen, 
     .                attoutflag, miss_in)
	ELSE
	   attname = '_FillValue'
	   CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	   IF (attid .GT. 0) THEN
	      no_fill_flag = 0
	      CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, attname, attype, attlen, attoutflag, status )
	      IF (attoutflag .EQ. 0) no_fill_flag = 1
	   ENDIF
	ENDIF

	IF (no_missing_flag .EQ. 1) THEN
	  attname = '_FillValue'
	  CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	  IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .                attid, attname, attype, attlen, attoutflag, status )
	   IF (attoutflag .EQ. 0) no_fill_flag = 1
	ENDIF

* If the data is numeric, check the missing flag, getting the original one, 
* and write with the output type

	IF (status .EQ. ferr_ok .AND. outtype .NE. nf_char) THEN

* If un-scaling on output, write the original missing flag.
	   IF (scalefac .NE. 1.D0  .OR. addoff .NE. 0.D0) THEN
	      vals(1) = miss_in(1)
	      vals(1) = vals(1)

* Otherwise convert missing flag to requested output type for the variable
	   ELSE
	      IF (attype .NE. outtype .AND. status .EQ. ferr_ok)  THEN
	         alen = TM_LENSTR1(typnam(attype))
	         slen = TM_LENSTR1(typnam(outtype))
	         IF ((.NOT.itsa_uvar) .AND. (dset .GT. 0 .AND. do_warn)) 
     .	          CALL WARN ('Converting data type of missing_value '//
     .	           typnam(attype)(1:alen)//' to match output type of variable '//
     .	           typnam(outtype)(1:slen) )
	      ENDIF
	   ENDIF
	   attype = outtype
  
	   IF (attoutflag .EQ. 1) THEN
	      IF (no_missing_flag.EQ.0) THEN
	         CALL CD_WRITE_ATTVAL_DP(
     .	                    cdfid, vbuff, 'missing_value', vals, attlen, 
     .	                    attype, status )
	         IF (status .NE. ferr_ok) GOTO 5000
	         got_missing = .TRUE.

	         IF (dset .EQ. pdset_uvars) THEN 
	            CALL CD_WRITE_ATTVAL_DP(
     .	              cdfid, vbuff, '_FillValue', vals, attlen, 
     .	              attype, status )
	            IF (status .NE. ferr_ok) GOTO 5000
	            got_fill = .TRUE.
	         ENDIF
	      ENDIF

* Ferret doesn't keep the values corresponding to two flags in the data that it stores.
* If there was a _FillValue in the input file, and its the bad-flag, write it.
* Issue #1883: If there is a _FillValue and it's NaN, it will be different from the
* bad-flag. We do want to write it.

	      IF (.NOT.got_fill ) THEN 
	         attname = '_FillValue'
	         CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
	         IF (attid .GT. 0) 
     .	            CALL CD_GET_VAR_ATT_INFO (dset, varid, 
     .	            attid, attname, attype, attlen, attoutflag, status )
	         IF (attoutflag .EQ. 0) no_fill_flag = 1
	         got_it = .FALSE.
	         IF (attoutflag .NE. 0) got_it = NC_GET_ATTRIB_DP
     .	            (dset, varid, attname, do_warn, vbuff, attlen, 
     .	             attoutflag, fill_in)
	         IF ( TM_CHECK_NAN(fill_in(1)) .EQ. 1   ) fill_in(1) = bad
	         IF (got_it .AND. (fill_in(1).EQ.bad .OR. fill_in(1).EQ.miss_in(1)) ) THEN
	            vals(1) = fill_in(1)
	            vals(1) = vals(1)
	            attype = outtype
	         ELSE
		    IF (its_cdf .AND. .NOT.itsa_uvar) no_fill_flag = 1
	         ENDIF
	      ENDIF
	      IF (no_fill_flag .EQ. 0) THEN
		 CALL CD_WRITE_ATTVAL_DP(
     .	           cdfid, vbuff, '_FillValue', vals, attlen, 
     .	           at type, status )
		 got_fill = .TRUE.	  
	      ENDIF

	   ENDIF
	ENDIF ! status 


* Get vbuff into original upper/lowercase form. If mode upcase_output is set, 
* then upcase the variable name.

	CALL CD_GET_VAR_INFO (dset, varid, buff, vtype, 
     .	  nvdims, vdims, nvatts, coordvar, all_outflag, status) 
	vbuff = buff
	IF (mode_upcase_output) CALL STR_UPCASE( vbuff, buff)

	DO 200 iatt = 1, nvatts
	   CALL CD_GET_VAR_ATT_NAME( dset, varid, iatt, attname, status)
	   CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt, attname, 
     .	     attype, attlen, attoutflag, status )
	        
	   slen = TM_LENSTR1(attname)
	   IF ( MATCH_NAME (attname,  slen, 'MISSING_VALUE', 13 ) .OR. 
     .	        MATCH_NAME (attname,  slen, '_FILLVALUE', 10 ) ) CYCLE   ! done already

	   IF ( MATCH_NAME (attname,  slen, 'SCALE_FACTOR', 12 ) .OR. 
     .	        MATCH_NAME (attname,  slen, 'ADD_OFFSET', 10 ) ) THEN
	      got_it = NC_GET_ATTRIB_DP (dset, varid, attname,
     .	             do_warn, vbuff, attlen, attoutflag, vals)
	      IF (attoutflag  .EQ. 1) CALL CD_WRITE_ATTVAL_DP (cdfid, vbuff,
     .	         attname, vals, attlen, attype, status )
	         IF (status .NE. ferr_ok) GOTO 5000
              CYCLE
	   ENDIF

	   attype = 0
	   attlen = 0
	   CALL CD_GET_VAR_ATT_INFO (dset, varid, iatt,
     .	         attname, attype, attlen, attoutflag, status )
     
	   IF (attype .NE. nf_char .AND. attlen .GT. 100) THEN
	      attlen = 100
	      alen = TM_LENSTR1(attname)
	      IF (do_warn) CALL WARN 
     .           ('writing only first 100 elements of attribute'//attname(1:alen))
	   ENDIF

	   IF (attype.EQ.nf_char .AND. attoutflag.EQ.1) THEN
	      got_it = NC_GET_ATTRIB ( dset, varid, attname, do_warn, vbuff, 
     .	            maxlen, attlen, attoutflag, buff, vals)
	      lbuff = TM_LENSTR1(buff)
	      CALL CD_WRITE_ATTRIB(cdfid, vbuff, attname, buff(:lbuff), .FALSE., status )

	   ELSE IF (attoutflag.EQ.1) THEN
	      got_it = NC_GET_ATTRIB ( dset, varid, attname, do_warn, vbuff, 
     .	            maxlen, attlen, attoutflag, buff, vals)
	      CALL CD_WRITE_ATTVAL_DP (cdfid, vbuff,attname, vals, attlen, attype, status )
	      IF (status .NE. ferr_ok) GOTO 5000

	   ENDIF

	   IF (MATCH_NAME (attname,  slen, 'LONG_NAME', 9 )) THEN
	      got_title = .TRUE.
	      output_title = (attoutflag .EQ. 1)
	   ENDIF

	   IF (MATCH_NAME (attname,  slen, 'UNITS', 5 )) THEN
	      got_units = .TRUE.
	      output_units = (attoutflag.EQ.1)
	   ENDIF

	   IF (MATCH_NAME (attname,  slen, 'HISTORY',  7 )) THEN
	      got_history = .TRUE.
	      output_history = (attoutflag.EQ.1)
	   ENDIF

 200	CONTINUE

! done getting attributes from linked-list structure

 5000	RETURN
	END
