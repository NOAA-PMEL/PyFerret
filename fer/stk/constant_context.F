	SUBROUTINE CONSTANT_CONTEXT
     .		      ( const_text, 
     .			default_cx,
     .			const_cx,
     .			status )

* return in const_cx context information describing the constant

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 3/28/87
* revision 0.1 - 5/5/88  - FINISH_CONTEXT_MODS --> FLESH_OUT_CONTEXT
* V200:  6/20/89 - re-worked user defined variable logic 
* V300: ifdef to work around "*" bug in READ (buff,*) in DEC f77 3.2.2
* V420 (IBM/AIX port of V411): 10/5/95 - buffer "READ(var,*"
	
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xalgebra.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XALGEBRA.CMN'
#endif

* calling argument declarations:
	INTEGER		default_cx, const_cx, status
	CHARACTER*(*)	const_text

* internal variable declarations:
	INTEGER		mods_cx, idim
	REAL		const_buff
	CHARACTER	buff*24

* equivalences
	REAL		cx_value( max_context )
	EQUIVALENCE   ( cx_variable, cx_value )

* initialize
	CALL TRANSFER_CONTEXT ( default_cx, const_cx )
	cx_category( const_cx ) = cat_constant
	cx_bad_data( const_cx ) = bad_val4
	cx_data_set( const_cx ) = pdset_irrelevant
	cx_dset_gvn( const_cx ) = .TRUE.

* get the value of the constant
* bug is that if a "*" follows the numeric text the error branch is taken
        buff = const_text
	READ ( buff, *, ERR=5100 ) const_buff
	cx_value( const_cx ) = const_buff

* flesh out context by removing all axes limits
	DO 200 idim = 1, 4
 200	CALL DEL_CX_DIM( idim, const_cx )

	RETURN

* error exit
 5000	RETURN
 5100	CALL ERRMSG( ferr_syntax, status, const_text, *5000 )
	END
