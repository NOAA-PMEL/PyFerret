	SUBROUTINE STRING_CONTEXT
     .		      ( uvar,
     .			item, 
     .			default_cx,
     .			string_cx,
     .			status )

* return in string_cx context information describing the string context
* this is for a "cat_string" - a literal string embedded in a var defn

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V500  1/99 *sh*  based on const_context
* V530  8/00 *sh*  adding support for arrays of strings
	
#ifdef unix
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include 'xfr_grid.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, string_cx, uvar, item, status

* internal variable declarations:
	INTEGER		idim

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, string_cx )
	cx_category( string_cx ) = cat_string
	cx_type    ( string_cx ) = ptype_string
	cx_bad_data( string_cx ) = bad_val4
	cx_data_set( string_cx ) = pdset_irrelevant
	cx_dset_gvn( string_cx ) = .TRUE.
	cx_variable( string_cx ) = 1000*uvar + item     ! packed info
	cx_grid    ( string_cx ) = mgrid_abstract

* flesh out context by removing all axes limits
	DO 200 idim = 1, 4
 200	CALL DEL_CX_DIM( idim, string_cx )

* set up the context to represent an array on the X (abstract) axis
!	cx_lo_ss(string_cx, x_dim) = 1
!	cx_hi_ss(string_cx, x_dim) = 1
!	cx_by_ss(x_dim, string_cx) = .TRUE.
!	CALL FLESH_OUT_AXIS( x_dim, string_cx, status )
!        ! status not reported -- no error possible

	status = ferr_ok
	RETURN

* error exit
	END
