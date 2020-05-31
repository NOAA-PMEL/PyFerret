	SUBROUTINE CDF_SET_OUT_TYPE (dset, varid, itsa_uvar, 
     .                itsa_string, out_type, outtype, status)


        include 'netcdf.inc'
        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xprog_state.cmn'
	include 'errmsg.parm'

	LOGICAL itsa_uvar, itsa_string
	INTEGER dset, varid, outtype, status
	CHARACTER*(*) out_type

	INTEGER TM_LENSTR1, vartype, slen, alen, attid
	CHARACTER attname*128, dum*10
	
* For error messages
      CHARACTER*9 typnam(6)
      DATA typnam/'NC_BYTE', 'NC_CHAR', 'NC_SHORT', 'NC_INT', 
     .            'NC_FLOAT', 'NC_DOUBLE'/

* Set the type for writing data to netCDF. 
* Apply a command-line output type if given, e.g. SAVE/OUTTYPE=INT
* Otherwise use an OUTTYPE atribute if it's been set, or use the data type
* from the source dataset.
* v7.6 *acm* 5/20 Issue 1876: can read string-typed data, write as char.
*                 Write a note but only if they've asked for /OUTTYPE=STR

* dset         source dataset
* varid        variable id if source is a netCDF file 
* itsa_uvar
* itsa_string  for expressions can't determine this from the dataset
* out_type     string from SAVE/OUTTYPE=
* outtype      return this netcdf type code
*
* To simply reset an output type according to the setting in OUT_TYPE, send in
* the variable type in "outtype" and set varid = -1 (needed in cdf_list_dsg)

* initialize
	status = ferr_ok

	IF (varid .GT. 0) THEN
	   CALL CD_GET_VAR_TYPE (dset, varid, dum, vartype, status)

	   IF (itsa_uvar) THEN  ! for user vars type not known in attrib structure
	      vartype = nf_double  
	      IF (itsa_string) vartype = nf_char
	   ENDIF
	   outtype = vartype

* If there is an OUTTYPE attribute, that came from a SET VAR/OUTTYPE.
* It will be superceded by a SET LIST/OUTTYPE or a SAVE/OUTTYPE setting,
* applied just below.

	   CALL CD_GET_VAR_OUTTYPE (dset, varid, outtype, status)

* If its just an expression it doesnt have a varid in the attribute structure
* for user-vars. Still set the output type.

	ELSE IF (varid .EQ. 0) THEN
	   vartype = 6  ! netcdf nf_double
	   IF (itsa_string) vartype = 2
	ELSE

* if not from a varid, just converting a type using the specified out_type
	   vartype = outtype  
	ENDIF

	IF (outtype .EQ. 0) outtype = vartype

* Setting made with SAVE/OUTTYPE or SET LIST/OUTTYPE
* SAVE/OUTTYPE takes precedence over a SET LIST/OUTTYPE
* out_type comes in from a SAVE/OUTTYPE  Use it if its not DFLT
* list_outtype is from SET LIST/OUTTYPE  IF out_type is DFLT, use list_outtype

	IF ( out_type(1:3) .EQ. 'DFL') out_type = list_outtype

	IF (out_type(1:3) .EQ. 'DOU') outtype = nf_double
	IF (out_type(1:3) .EQ. 'FLO') outtype = nf_float
	IF (out_type(1:3) .EQ. 'INT') outtype = nf_int
	IF (out_type(1:3) .EQ. 'SHO') outtype = nf_short
	IF (out_type(1:3) .EQ. 'BYT') outtype = nf_byte
	IF (out_type(1:3) .EQ. 'STR') THEN
	   outtype = nf_char
	   CALL WARN ('Writing with type STRING not implemented. Using type CHAR.')
	ENDIF
	IF (outtype .EQ. nf_string) outtype = nf_char

	IF (dset .GT. pdset_irrelevant .AND.
     .      outtype.NE.nf_char .AND. outtype.NE.nf_string .AND.
     .      outtype.GT.nf_double) outtype = nf_double

* When the user requests higher precision output from a lower precision file 
* variable, e.g. converting FLOAT to DOUBLE, write an attribute explaining 
* that it contains data converted in this way.

	IF (dset .GT. pdset_irrelevant .AND. outtype.GT.vartype .AND.
     .      outtype.NE.nf_char .AND. outtype.NE.nf_string) THEN

           attname = 'Ferret_Precision_Note'
           CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status)
           IF (status.EQ.ferr_ok) 
     .             CALL CD_DELETE_ATTRIBUTE(dset, varid, attname, status)
           slen = TM_LENSTR1(typnam(vartype))
           alen = TM_LENSTR1(typnam(outtype))
           CALL CD_PUT_NEW_ATTR (dset, varid, 
     .              attname, 2, 61, 1, 
     .         'This variable written using Ferret was converted from '//
     .              typnam(vartype)(4:slen)//' to '//
     .              typnam(outtype)(4:alen), 1., status )

	ENDIF

	RETURN
	END
