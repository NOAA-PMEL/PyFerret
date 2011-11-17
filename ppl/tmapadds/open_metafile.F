        SUBROUTINE OPEN_METAFILE

*
* Metafiles no longer supported.
* All graphics files are now created by the graphics delegate.
*
        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'

*       Set meta_lun to non-zero to indicate that graphics
*       should be saved to meta_file at the appropriate time.
        meta_lun = 1

        RETURN
        END

