	SUBROUTINE GET_DSG_MASK_TITLE (dset, title, has_mask, slen)

	include 'errmsg.parm'

	LOGICAL has_mask
	INTEGER dset, slen
	CHARACTER*(*) title
	
	LOGICAL NC_GET_ATTRIB_STRING
	INTEGER varid, attid, maxlen, attoutflag, status
	CHARACTER v*1, attname*32

* Initialize 

	has_mask = .FALSE.
	slen = 0

	attname = '__feature_mask_'
	varid = 0
	v = '.'

	CALL CD_GET_VAR_ATT_ID (dset, varid, attname, attid, status) 
	IF (attid .LE. 0 .OR. status .NE. ferr_ok) GOTO 5010

	maxlen = 128
	has_mask = NC_GET_ATTRIB_STRING ( dset, varid, attname,
     .                                   .FALSE., v, maxlen,
     .                                   slen, attoutflag, title)

 5000	RETURN
 5010	slen = 0

	END
