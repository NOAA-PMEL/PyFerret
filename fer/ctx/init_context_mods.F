	SUBROUTINE INIT_CONTEXT_MODS ( cx )

* prepare a clean,empty cx to receive a list of modifications

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - ????
* revision 0.1 -  4/1/89  - added regridding transformations
* V200:  6/28/89 - 4D symmetrical - init trans_arg, too, eliminate %%_relative
*		 - init cx_dset_gvn, cx_grid_needs_dset
* V420: 9/8/95 - added initializations for implicit grid creation
* 
*       8/97 *kob* - initalize cx_calendar - used to indicate whether a date
*                    contains a string 
#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations:
	INTEGER	cx

* internal variable declarations:
	INTEGER	idim

* initialize everything but region
	cx_variable( cx ) = unspecified_int4
	cx_category( cx ) = unspecified_int4
	cx_data_set( cx ) = unspecified_int4
	cx_dset_gvn( cx ) = .FALSE.
	cx_grid( cx )     = unspecified_int4
	cx_unstand_grid( cx ) = .FALSE.
	cx_grid_needs_dset    = .FALSE.
	cx_has_impl_grid( cx ) = .FALSE.
	cx_calendar(cx) = .FALSE.

* initialize region
* note: ss specifications are not used in storing modifications
	DO 100 idim = 1,4
	   cx_lo_ss ( cx, idim )  = unspecified_int4
	   cx_hi_ss ( cx, idim )  = unspecified_int4
	   cx_lo_ww ( idim, cx )  = unspecified_val8
	   cx_hi_ww ( idim, cx )  = unspecified_val8
	   cx_delta ( idim, cx )  = unspecified_val8
	   cx_trans ( idim, cx )  = trans_no_transform
	   cx_trans_arg(idim,cx)  = bad_val4
	   cx_by_ss ( idim, cx )  = .FALSE.		!for bookkeeping
	   cx_given( idim, cx )  = .FALSE.
	   cx_regrid_trans(idim,cx) = unspecified_int4 
100	CONTINUE
	RETURN
	END
