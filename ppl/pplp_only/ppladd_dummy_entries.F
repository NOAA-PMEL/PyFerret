	SUBROUTINE DUMMY_ENTRIES

* 	Entry points that are meaningful in FERRET but no-ops for PPLUS
*	INTERRUPT actually does something so it's not really a dummy -

#ifdef unix
        include         'tmap_pplv11inc/xinterrupt.cmn' ! 1/89 *sh*
        include         'tmap_pplv11inc/ppl_in_ferret.cmn'
#else
        INCLUDE         'TMAP_PPLV11INC:XINTERRUPT.CMN' ! 1/89 *sh*
        INCLUDE         'TMAP_PPLV11INC:PPL_IN_FERRET.CMN'
#endif

	REAL xorg, yorg, size, scale, xclip, yclip, xll, yll

********************************************************************************

	ENTRY SEG_ON( )

	ENTRY SEG_OFF( )

	ENTRY GET_VP_FROM_FERRET(  xorg, yorg,
     .				   size, scale, xclip, yclip, xll, yll )

	ENTRY CLEAR_VP

        ENTRY ERRMSG

        ENTRY GKS_X_CONID

        ENTRY XGT_GRID_DATA

	ENTRY BOX_BOUNDS

        ENTRY GET_VIEW_SIZE

	ENTRY CK_GKS_ACTIVE

        ENTRY FERRET_PLOT_COMPLETE

	STOP 'Attempt to call FERRET routine from PPLP'

	ENTRY INTERRUPT
	interrupted = .false.

	END
