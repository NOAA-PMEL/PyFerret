	LOGICAL FUNCTION CX_DIM_DIFF (cx1, cx2, idim)

* Determine whether the axis contexts differ for two contexts
* in a given direction.
*
* Ansley Manke 9/2020
*

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
        include 'xcontext.cmn'

	INTEGER cx1, cx2, idim

	CX_DIM_DIFF = .FALSE.
	
	IF ( cx_lo_ss(cx1, idim) .NE. cx_lo_ss(cx2, idim) ) CX_DIM_DIFF = .TRUE.
	IF ( cx_hi_ss(cx1, idim) .NE. cx_hi_ss(cx2, idim) ) CX_DIM_DIFF = .TRUE.
	IF ( cx_lo_ww(idim, cx1) .NE. cx_lo_ww(idim, cx2) ) CX_DIM_DIFF = .TRUE.
	IF ( cx_hi_ww(idim, cx1) .NE. cx_hi_ww(idim, cx2) ) CX_DIM_DIFF = .TRUE.
	   
	RETURN
	END
