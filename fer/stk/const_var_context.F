	SUBROUTINE CONST_VAR_CONTEXT
     .		      ( text,
     .			uvar,
     .			item, 
     .			default_cx,
     .			const_cx,
     .			status )

* return in const_cx context information describing the constant array

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V5.00: 12/18/98 - based on CONSTANT_CONTEXT	
*	 2/99 *sh* - bug fix on cx_given for normal axes
* V530:  8/00 *sh* - allow constant arrays of strings
* V533:  7/01 *sh* - allow single quoted strings

	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xcontext.cmn'
	include	'xalgebra.cmn'
	include 'xfr_grid.cmn'

* calling argument declarations:
	INTEGER		uvar, item, default_cx, const_cx, status
	CHARACTER*(*)	text

* internal variable declarations:
	INTEGER		idim

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, const_cx )
	cx_category( const_cx ) = cat_const_var
	cx_bad_data( const_cx ) = bad_val4
	cx_data_set( const_cx ) = pdset_irrelevant
	cx_dset_gvn( const_cx ) = .TRUE.
	cx_variable( const_cx ) = 1000*uvar + item     ! packed info
	cx_grid    ( const_cx ) = mgrid_xabstract

* identify a constant array of strings by the presence of a quotation mark
	IF ( INDEX(text,'"') .GT. 0
     .	.OR. INDEX(text,"'") .GT. 0 ) THEN
	   cx_type(const_cx) = ptype_string
	ELSE
	   cx_type(const_cx) = ptype_float
	ENDIF

* definition applies only to the axis of the constant list
	cx_given( x_dim, const_cx ) = .TRUE.
	DO 100 idim =  y_dim, t_dim
 100	cx_given( idim, const_cx ) = .FALSE.

* remove all axes limits
	DO 200 idim = y_dim, t_dim
 200	CALL DEL_CX_DIM( idim, const_cx )

* set the X axis limits equal to the length of the array
	cx_lo_ss(const_cx, x_dim) = 1
	CALL COUNT_NUMBER_LIST(text, cx_hi_ss(const_cx,x_dim) )
	cx_by_ss(x_dim,const_cx) = .TRUE.
	CALL FLESH_OUT_AXIS(x_dim, const_cx, status)

	RETURN

!* error exit
! 5000	RETURN
! 5100	CALL ERRMSG( ferr_syntax, status, const_text, *5000 )
	END
