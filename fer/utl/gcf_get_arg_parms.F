	SUBROUTINE GCF_GET_ARG_PARMS
     .			(gcfcn, iarg, lo_off,  hi_off, implied_lims)

* Return parameters about the grid changing function, gcfcn, on how the
* limits of the argument, iarg, should be determined relative to the
* default limits of the grid changing variable

* For example, in
*	LET a = GCFCN( b )
*	LIST a[i=1:5]
* we need to know if the limits i=1:5 should be passed along to b and if they
* should be increased or decreased by offsets supplied by the function.

* V450: 1/97
*       97.07.16 *jc* adding code to query external functions

* calling argument declarations
	LOGICAL	implied_lims(4)
	INTEGER	gcfcn, iarg, lo_off(4),  hi_off(4)

* internal variable declarations
	INTEGER	idim, iptr, ip

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xtm_grid.cmn_text'
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xgrid_chg_fcns.cmn'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XGRID_CHG_FCNS.CMN'
#endif

* check the internal grid changing functions first
        IF (gcfcn .LT. 0 ) THEN

           STOP 'gcf_get_arg_parms'

        ELSEIF (gcfcn .LE. gfcn_num_internal) THEN

* is this a valid argument number?
	   IF (iarg .LT. 1
     .    .OR. iarg .GT. gfcn_num_reqd_args(gcfcn) )
     .					STOP 'gcf_cx_merge_arg_1'

* determine location of information in argument arrays
	   iptr = gfcn_arg_ptr(gcfcn) + iarg - 1

* return information about argument iarg

	   DO 100 idim = 1, 4

	      lo_off(idim) = gfcn_arg_extend_lo(idim, iptr)
	      hi_off(idim) = gfcn_arg_extend_hi(idim, iptr)
	      implied_lims(idim) = gfcn_axis_implied_from(idim, iptr)
 100	   CONTINUE

	ELSE

	   call efcn_get_axis_extend_lo(gcfcn, iarg, lo_off)
	   call efcn_get_axis_extend_hi(gcfcn, iarg, hi_off)
	   call efcn_get_axis_implied_from(gcfcn, iarg, implied_lims)

	ENDIF

	RETURN
	END
