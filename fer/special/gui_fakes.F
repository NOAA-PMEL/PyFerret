! gui_fakes
! routines to make it possible to link FERRET without its GUI

	SUBROUTINE FERRET_LIST_IN_WINDOW( fhol, ftxout_append )

#ifdef unix
        include 'ferret_cmn/ferret.parm'
        include 'ferret_cmn/xprog_state.cmn'
#else
        INCLUDE 'FERRET_CMN:FERRET.PARM'
        INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER fhol, ftxout_append

	WRITE (err_lun, *) 
     .		'FERRET_LIST_IN_WINDOW called from non-GUI version'

	RETURN
	END

****************************************************************

	SUBROUTINE FERRET_WARN_IN_WINDOW( fhol, ftxout_append )

#ifdef unix
        include 'ferret_cmn/ferret.parm'
        include 'ferret_cmn/xprog_state.cmn'
#else
        INCLUDE 'FERRET_CMN:FERRET.PARM'
        INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
#endif

* calling argument declarations:
	INTEGER fhol, ftxout_append

	WRITE (err_lun, *) 
     .		'FERRET_WARN_IN_WINDOW called from non-GUI version'

	RETURN
	END
