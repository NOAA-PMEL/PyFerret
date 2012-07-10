*
* Metafiles no longer supported.  All graphics files are now created
* by the graphics delegate.  This routine informs the graphics engine
* of the image filename given in meta_file and sets save_on_exit to
* .TRUE.
*
        SUBROUTINE OPEN_METAFILE
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0

        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'
        INCLUDE 'fgrdel.cmn'

*       If the image filename has already been sent, ignore this call.
        IF ( imgname_sent ) RETURN

*       Send the image filename to currently active window.
        CALL FGD_SETIMGNAME(meta_file, imgname_sent)

        IF ( imgname_sent ) THEN
*           If successful, clear the currently active window
            CALL FGD_GCLRWK(activewindow, 0)
*           Make sure the save_on_exit flag is set
            save_on_exit = .TRUE.
        ENDIF

        RETURN
        END

