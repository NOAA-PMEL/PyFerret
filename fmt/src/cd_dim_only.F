	SUBROUTINE CD_DIM_ONLY (cdfid, name, nlen, dim_only)

* include files

	include 'netcdf.inc'

	INTEGER cdfid, nlen
	LOGICAL dim_only
	CHARACTER*(*) name

	INTEGER cdfstat, dimid, axid


	dim_only = .FALSE.

	cdfstat = NF_INQ_DIMID( cdfid, name(:nlen), dimid)

* already defined ? (limited consistency checking)
	IF ( cdfstat .EQ. NF_NOERR ) THEN

* ... is there a coordinate variable by the same name ?
	   cdfstat = NF_INQ_VARID( cdfid, name(:nlen), axid )

* The dimension matches but there is no coord var. 
* This lets us append variables to a DSG file.

	   dim_only = cdfstat .EQ. nf_enotvar

	ENDIF

	RETURN
	END
