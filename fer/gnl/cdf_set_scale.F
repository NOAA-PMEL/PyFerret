	SUBROUTINE CDF_SET_SCALE (dset, varid, do_warn, scalefac, addoff, ferr_ok, status)

	INTEGER dset, varid, ferr_ok, status
	LOGICAL do_warn
	REAL scalefac, addoff


	LOGICAL NC_GET_ATTRIB, got_it
	INTEGER attid, attype, attlen, attoutflag, buff_size
	REAL vals

	CHARACTER attname*128, varcode*128, buff*128

	IF (varid .LE. 0) GOTO 5000

	buff_size = 128
	scalefac = 1.
	addoff = 0.
	
* If the variable is to be written scaled, set the scale and offset
*  values that will be written as attributes and sent to CD_WRITE_VAR. 

	      
	attname = 'scale_factor'
	CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, 
     .                  status)
	IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                  varid, attid, attname, attype, attlen, attoutflag, 
     .                  status )

	IF (status.EQ.ferr_ok .AND. attid.GT.0 .AND. attoutflag.EQ.1) THEN
	   got_it = NC_GET_ATTRIB (dset, varid, attname,
     .                  do_warn, varcode, buff_size, attlen, 
     .                  attoutflag, buff, vals)
	   	scalefac = vals
	ENDIF

	attname = 'add_offset'
	CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid,  status)
	IF (attid .GT. 0) CALL CD_GET_VAR_ATT_INFO (dset, 
     .                  varid, attid, attname, attype, attlen, attoutflag, 
     .                  status )

	IF (status.EQ.ferr_ok .AND. attid.GT.0 .AND. attoutflag.EQ.1) THEN
	   got_it = NC_GET_ATTRIB (dset, varid, attname,
     .                  do_warn, varcode, buff_size, attlen, 
     .                  attoutflag, buff, vals)
	   addoff = vals
	ENDIF

 5000	RETURN
	END
