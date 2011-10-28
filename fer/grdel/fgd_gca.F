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

        SUBROUTINE FGD_GCA(left, bottom, right, top, totcols, totrows,
     .                 stcol, strow, numcols, numrows, colorindices)

        IMPLICIT NONE
        INTEGER totcols, totrows, stcol, strow, numcols, numrows,
     .          colorindices
        REAL left, bottom, right, top
        DIMENSION colorindices(totcols * totrows)

        INCLUDE 'ferret.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

        INTEGER numcells, k, colornum, success, errmsglen
        CHARACTER*2048 errmsg
*       Have to hard-code max number of cell color objects (doubles)
*       since pplmem is floats (memory alignment issues)
        INTEGER maxcells
        PARAMETER(maxcells = 64 * 1024)
        REAL*8 cellcolors(maxcells)

*       Sanity checks
        IF ( (activewindow .LT. 1) .OR.
     .       (activewindow .GT. maxwindowobjs) ) THEN
            STOP 'FGD_GCA: No active window'
        ENDIF
        IF ( windowobjs(activewindow) .EQ. nullobj ) THEN
            STOP 'FGD_GCA: null active windowobj'
        ENDIF
        numcells = numrows * numcols
        IF ( numcells .GT. maxcells ) THEN
            STOP 'FGD_GCA: too many cells for cell color object array'
        ENDIF

*       Assign the array of color objects from the indices
        DO 10 k = 1, numcells
            colornum = colorindices(k) + 1
            IF ( (colornum .LT. 1) .OR.
     .           (colornum .GT. maxcolorobjs) ) THEN
                STOP 'FGD_GCA: invalid color index'
            ENDIF
            cellcolors(k) = colorobjs(colornum, activewindow)
            IF ( cellcolors(k) .EQ. nullobj ) THEN
                STOP 'FGD_GCA: null color object'
            ENDIF
  10    CONTINUE

*       Draw the multicolored rectangle
        CALL FGDDRAWMULTICOLOR(success, windowobjs(activewindow),
     .                         left, bottom, right, top,
     .                         numrows, numcols, cellcolors)
        IF ( success .EQ. 0 ) THEN
            errmsg = ' '
            CALL FGDERRMSG(errmsg, errmsglen)
            CALL SPLIT_LIST(pttmode_help, err_lun, errmsg, errmsglen)
        ENDIF

        RETURN
        END
