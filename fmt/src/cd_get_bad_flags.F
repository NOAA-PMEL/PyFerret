	SUBROUTINE CD_GET_BAD_FLAGS (dset, varid, ds_bad, ds_miss, 
     .                               bad_default, var, nsiz)


* Consolidate to one bad-value - needed for the fix to 1861...
	INTEGER dset, varid, nsiz
	REAL    ds_bad, ds_miss, bad_default, var(*)



	LOGICAL NC_GET_ATTRIB, do_warn, scaled,
     .		got_fill, got_bad, got_scale, got_offset
	INTEGER maxlen, attlen, attoutflag
	CHARACTER*10 dum_name, buff
	REAL	scale, offset

*  vname in the call below is only for error messages, used only if
*  do_warn is set.

	do_warn = .FALSE.
	dum_name = 'dummy'
	maxlen = 10

* Get the missing and fill value attributes
*   _FillValue --> ds_missing_flag(ivtm)
*   missing_value --> ds_bad_flag(ivtm)

	got_fill = NC_GET_ATTRIB( dset, varid, '_FillValue',
     .                     do_warn, dum_name, maxlen, attlen, 
     .                     attoutflag, buff, ds_miss)
	got_bad = NC_GET_ATTRIB( dset, varid, 'missing_value',
     .                     do_warn, dum_name, maxlen, attlen, 
     .                     attoutflag, buff, ds_bad)

	IF (got_bad  .AND. .NOT.got_fill) ds_miss = ds_bad
	IF (got_fill .AND. .NOT.got_bad ) ds_bad = ds_miss

* If neither is set, use NaN
	IF (.NOT.got_fill .AND. .NOT.got_bad) THEN
	   CALL SET_NAN (ds_bad)
	   CALL SET_NAN (ds_miss)
	   CALL SWITCH_NAN(ds_miss, ds_bad, bad_default)
	ELSE

* Call SWITCH_NAN to make sure if NaN exists, its in ds_missing_flag
	   CALL SWITCH_NAN(ds_bad, ds_miss, bad_default)
	ENDIF

* Consolidate to one bad-value in the data

	CALL TM_SWITCH_BAD ( ds_miss, ds_bad, var, nsiz )

* Are there scale and offset attributes?

	maxlen = 2
	scale = 1.
	got_scale = NC_GET_ATTRIB( dset, varid, 'scale_factor',
     .	          do_warn, dum_name, maxlen, attlen, 
     .	          attoutflag, buff, scale)

	offset = 0.
	got_offset = NC_GET_ATTRIB( dset, varid, 'add_offset',
     .	          do_warn, dum_name, maxlen, attlen, 
     .	          attoutflag, buff, offset)

	IF ( got_scale .OR. got_offset) CALL TM_SCALE_SWITCH_BAD (
     .                   scale, offset, ds_bad, bad_default, var, nsiz )


	RETURN
	END
