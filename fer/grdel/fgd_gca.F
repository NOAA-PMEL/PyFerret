*
* Draw a multicolored rectangle
* (fill cell array)
*
* The rectangle from (left, bottom) to (top, right) is filled with
* colors from colorindices.  The arguments totcols and totrows are
* only used for dimensioning colorindices (so basically ignored).
* The arguments igcol and igrow are ignored here as they were also
* ignored in the GKS code.  (They were suppose to be the start indices.)
* The rectangle is divided evenly into numcols columns (X-axis) and
* numrows rows (Y-axis) and filled from colors in colorindices assign
* in column-major order.
*
* Not working properly and not actually needed, so stubbed out.
* Modern graphics engines would turn this into multiple calls to
* drawRectangle anyway, so just have Ferret make the drawRectangle
* calls instead.
*

        SUBROUTINE FGD_GCA(left, bottom, right, top, totcols, totrows,
     .                 stcol, strow, numcols, numrows, colorindices)

        IMPLICIT NONE
        INTEGER totcols, totrows, stcol, strow, numcols, numrows,
     .          colorindices
        REAL left, bottom, right, top

        STOP 'FGD_GCA: Stubbed, unexpected call'

        RETURN
        END
