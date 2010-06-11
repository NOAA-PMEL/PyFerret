      SUBROUTINE ADD_RDC_XFORM ( cx, status )

* add a hidden @RDC transform if appropriate

* programmer - jonathan callahan
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* version 5.0 - 10/20/98

* The @RDC ("reduced") transform is an attempt to use Ferret's
* transform capabilities to create a result context which is 
* of different dimensionality than the argument contexts while
* passing result context information upstream to the arguments.
*
* Specifically, we are working with external functions where
* we would like to have the result be a flag indicating success.
* A simple example is the writev5d.F external function which
* writes a vis5D file for some arguments over a set region.
* Our result should be a one dimensional flag.  However, we 
* would still like to be able to use SET REGION to specify the 
* a region which should be applied to the arguments.  Normally,
* SET REGION only applies to the result context so that we
* currently, before this fix, have this behavior:
*
*	SET REGION/x180:160w/y=20s:20n/l=1
*
*	let a = writev5d(sst, airt)
*	list a                          ! Lists 180 X 90 X 12 values
* 
* The workaround is the ugly:
*
*   let a = writev5d(sst[x=180:160w,y=20s:20n,l=1], airt[...])
*
*
* The @RDC transform is applied internally in the grid 
* determination phase and is hidden from the user.  The 
* program structure is:
*
* interp_stack.F
*        |
*      grid
*  determination -- get_uvar_grid.F -- is_uvar_grid.F -- get_var_context.F
*      phase
*        |
*        |
*
*
* This subroutine needs to apply the @RDC transform whenever
* the result axis inherits from an argument axis but the
* result axis is reduced.
* This is really a sneaky way of saying that the argument context
* depends on the result context (ie SET REGION information).
*
* Places where changes where made to accomodate @RDC:
*
* ferret.parm         -- pgc_span_*, trans_reduced
* xgrid_chg_fcns.parm -- gfcn_reduction(4,pmax_grid_chg_fcns)
* interp_stack.parm   -- isact_reduce
* EF_Util.h           -- axis.reduction
* EF_InternalUtil.c   -- efcn_get_axis_reduction_(), EF_New()
* get_var_context.F   -- CALL ADD_RDC_XFORM()
* gcf_impose_axes.F
* interp_stack.F
*


#ifdef unix
      include 'tmap_dims.parm'
      include 'xdset_info.cmn_text'
      external xdset_info_data
      include 'ferret.parm'
      include 'xcontext.cmn'
      include 'xgrid_chg_fcns.cmn'
      include 'errmsg.parm'
#else 
      INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
      INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
      INCLUDE 'FERRET_CMN:FERRET.PARM'
      INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
      INCLUDE 'FERRET_CMN/XGRID_CHG_FCNS.CMN'
      INCLUDE 'FERRET_CMN/ERRMSG.PARM'
#endif
      
      LOGICAL ITSA_GCVAR
      INTEGER WHICH_GCFCN

      LOGICAL valid_limits
      INTEGER cx, status, uvar, gcfcn, idim
      INTEGER axis_reduction(4)
      
      valid_limits = .TRUE.

      IF ( cx_category(cx) .EQ. cat_user_var ) THEN

         uvar = cx_variable(cx)
         IF ( ITSA_GCVAR(uvar) ) THEN

            gcfcn = WHICH_GCFCN(uvar)

            IF ( gcfcn .GT. gfcn_num_internal ) THEN

               CALL EFCN_GET_AXIS_REDUCTION(gcfcn, axis_reduction)
               DO 100 idim = 1, 4

* Check that we have valid limits
                  IF ( cx_by_ss(idim,cx) ) THEN
                     IF ( cx_lo_ss(cx,idim) .EQ. unspecified_int4 
     .                 .OR. cx_hi_ss(cx,idim) .EQ. unspecified_int4 )
     .                 THEN
                          RETURN
                     ENDIF
                  ELSE
                     IF ( cx_lo_ww(idim,cx) .EQ. unspecified_val8 
     .                 .OR. cx_hi_ww(idim,cx) .EQ. unspecified_val8 )
     .                 THEN
                          RETURN
                     ENDIF
                  ENDIF

* Check that we have a valid grid or else FLESH_OUT_AXIS will fail
                  IF ( cx_grid(cx) .EQ. unspecified_int4 ) THEN
                     RETURN
                  ENDIF

                  IF ( axis_reduction(idim) 
     .                 .EQ. pgc_span_reduced ) THEN
                     cx_trans    ( idim, cx ) = trans_reduced
                     cx_trans_arg( idim, cx ) = bad_val4
                     CALL FLESH_OUT_AXIS(idim, cx, status)
                  ENDIF 

 100           CONTINUE

            ENDIF

*     Nothing to do YET for internal gc functions

         ENDIF

*     Nothing to do for non-gc functions
         
      ENDIF

*     Nothing to do for non-user variables
         
*     No way to return non-ok status yet.

      status = ferr_ok

      RETURN

* error exit

* 5100   CALL ERRMSG( ferr_syntax,status,'illegal name: '//buff3,*5000 )

      END
      
