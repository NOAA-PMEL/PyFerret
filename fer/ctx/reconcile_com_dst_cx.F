	SUBROUTINE RECONCILE_COM_DST_CX( com_cx, dst_cx, protected_ax )

* The component context is fully fleshed out (necessarily, since
* the component has already been obtained) whereas the destination (result)
* context may still have some unknown axis limits.  So pass the component
* information over to the destination on all except the critical axis
* on which the transformation/regrid, etc is about to be performed

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V420: 11/95 - code lifted from IS_REGRID **including warnings about untested
*			alternatives**

#ifdef unix
	include	'ferret_cmn/ferret.parm'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
#endif

* calling argument declarations:
	INTEGER	com_cx, dst_cx, protected_ax

* Make the destination context like the source data context on all except
* the regrid axis.  That one keeps the user-requested size.
* (Other axes may have been modified by information embedded within a
* user-variable definition.  An alternative (untested 8/89) approach is to
* call PASS_AMBIG_LI** for uvars)

	CALL TRANSFER_CONTEXT( com_cx, cx_buff )

	CALL TRANSFER_AXIS( protected_ax, dst_cx, cx_buff )

	CALL TRANSFER_CONTEXT( cx_buff, dst_cx )

	RETURN
	END	
