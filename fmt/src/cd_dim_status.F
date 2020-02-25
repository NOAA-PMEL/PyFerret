	SUBROUTINE CD_DIM_STATUS (cdfid, name, nlen, dim_status)


* dim_status = 0 if dimension does not yet written
*            = 1 if dimension exists but there is not a coordinate variable
*            = 2 if dimension and coordinate variable are already in the file

* include files

	include 'netcdf.inc'

	INTEGER cdfid, nlen, dim_status
	CHARACTER*(*) name

	INTEGER cdfstat, dimid, axid

	dim_status = 0

	cdfstat = NF_INQ_DIMID( cdfid, name(:nlen), dimid)

* already defined ? (limited consistency checking)
	IF ( cdfstat .EQ. NF_NOERR ) THEN

* ... is there a coordinate variable by the same name ?
	   cdfstat = NF_INQ_VARID( cdfid, name(:nlen), axid )

	   IF ( cdfstat .EQ. NF_NOERR ) THEN
	      dim_status = 1 
	   ELSE

* The dimension matches but there is no coord var. 
* This lets us append variables to a DSG file.

	      IF (cdfstat .EQ. nf_enotvar) dim_status = 1
	   ENDIF

* other cases, dim_status remains 0.  What else could there be? ...

	ENDIF

	RETURN
	END
