	SUBROUTINE INIT_EMPTY_CONTEXT ( cx )

* set up a "null" context suitable to merge other contexts on top of

* V450: 1/97 - part of changes to accomodate grid-changing functions
* 9/97 *kob* - initialize cx_calendar as well
#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER cx

* internal variable declarations
	INTEGER	idim

* all contexts contain axis region information
	DO 100 idim = 1,4
	   cx_lo_ss ( cx, idim )   = unspecified_int4
	   cx_hi_ss ( cx, idim )   = unspecified_int4
	   cx_lo_ww ( idim, cx )   = unspecified_val8
	   cx_hi_ww ( idim, cx )   = unspecified_val8
	   cx_trans ( idim, cx )   = trans_no_transform
	   cx_by_ss ( idim, cx )   = .TRUE.
	   cx_given ( idim, cx )   = .TRUE.		! <== (*)
	   cx_trans_arg(idim, cx)  = bad_val4
	   cx_delta ( idim, cx )   = unspecified_val8
	   cx_regrid_trans(idim,cx)=unspecified_int4
 100	CONTINUE

	cx_data_set( cx ) = unspecified_int4
	cx_dset_gvn( cx ) = .TRUE.			! <== (*)
	cx_category( cx ) = unspecified_int4
	cx_variable( cx ) = unspecified_int4
	cx_grid    ( cx ) = unspecified_int4
!	cx_ovsiz   ( cx ) = 
!	cx_bad_data( cx ) = 

	cx_unstand_grid ( cx ) = .FALSE.
	cx_impl_grid    ( cx ) = unspecified_int4

* kob 9/97 - need to initialize cx_calendar as well
		 IF (idim .eq. t_dim) 
     .                cx_calendar(cx) = .FALSE.

* (*) initialized to TRUE so that ANDing with another context will yield
*	TRUE only if "given" in that context

	RETURN
	END
