	SUBROUTINE GET_CONTEXT_BAD_FLAG ( cx )

* insert bad data flag into context for variable

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 4/1/87

#ifdef unix
	include	'tmap_format/tmap_dims.parm'
	include	'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE	'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

	INTEGER cx


	IF     ( cx_category( cx ) .EQ. cat_file_var ) THEN

	   cx_bad_data( cx ) = ds_bad_flag( cx_variable( cx ) )

	ELSE

	   cx_bad_data( cx ) = bad_val4

	ENDIF

	RETURN
	END
