        SUBROUTINE OPEN_METAFILE

*
* Metafiles no longer supported.
* All graphics files are now created by the graphics delegate.
*
        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'
        INCLUDE 'fgrdel.cmn'

*       Do not set meta_open to .True., since this tells Ferret
*       to also write graphics commands to meta_wsid for creating
*       the GKS metafile.  However, this means that this routine
*       will get called with every new plot.  So rely on meta_lun
*       instead.

*       If already open, ignore this call
        IF ( meta_lun .NE. 0 ) RETURN

*       Set the flag indicating the graphics should
*       be saved to file at the appropriate time.
        meta_lun = 1

*       Clear the currently active window
        CALL FGD_GCLRWK(activewindow, 0)

        RETURN
        END

