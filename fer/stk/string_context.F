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
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

        IMPLICIT NONE
	include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include 'xfr_grid.cmn'

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
	DO idim = 1, nferdims
	   CALL DEL_CX_DIM( idim, string_cx )
 	ENDDO

	status = ferr_ok
	RETURN

* error exit
	END
