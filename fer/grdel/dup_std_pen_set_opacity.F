      SUBROUTINE DUP_STD_PEN_SET_OPACITY(windowid, ipen, opacity_frac)

*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*

*  For the active window, uses the RGB and thickness part of the given 
*  standard pen and the given opacity fraction to create a new temporary 
*  pen.  The ID of this new pen is returned in ipen.  Prints an error
*  message if any problems arise.

        IMPLICIT NONE
        INCLUDE 'ferret.parm'
        INCLUDE 'tmap_dims.parm'
        INCLUDE 'xprog_state.cmn'
        INCLUDE 'fgrdel.cmn'

* calling argument declarations:
        INTEGER windowid, ipen
        REAL*4 opacity_frac

* local argument declarations:
        INTEGER ncolors, icolor, ier, errstrlen, TM_LENSTR
        REAL*4 dfltthick, redfrac, grnfrac, bluefrac, opaqfrac
        CHARACTER*256 errstr

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            errstr = 'DUP_STD_PEN_SET_OPACITY: Invalid windowid'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            errstr = 'DUP_STD_PEN_SET_OPACITY: null windowobj'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        ncolors = mode_arg(pmode_linecolors,1)
        IF ( (ipen .LT. 1) .OR. (ipen .GT. 3*ncolors) ) THEN
            errstr = 'DUP_STD_PEN_SET_OPACITY: invalid ipen'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF
        IF ( (opacity_frac .LT. 0.0) .OR. (opacity_frac .GT. 1.0) ) THEN
            errstr = 'DUP_STD_PEN_SET_OPACITY: Invalid opacity_frac'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Get the color and thickness of the given standard pen
        IF ( ipen .LE. ncolors ) THEN
           icolor = ipen
           dfltthick = 1.0
        ELSE IF ( ipen .LE. 2*ncolors ) THEN
           icolor = ipen - ncolors
           dfltthick = 2.0
        ELSE
           icolor = ipen - 2*ncolors
           dfltthick = 3.0
        ENDIF

*       Get the RGB for the color of the pen
        CALL FGD_GQCR(windowid, icolor, ier, 
     .                redfrac, grnfrac, bluefrac, opaqfrac)
        IF ( ier .NE. 0 ) THEN
            errstr = 'DUP_STD_PEN_SET_OPACITY: invalid icolor from ipen'
            errstrlen = TM_LENSTR(errstr)
            CALL SPLIT_LIST(pttmode_help, err_lun, errstr, errstrlen)
            RETURN
        ENDIF

*       Create a new temporary color with the new opacity
        opaqfrac = opacity_frac
        CALL FGD_CREATE_TEMP_COLOR(windowid, icolor, 
     .                  redfrac, grnfrac, bluefrac, opaqfrac)

*       Create a new temporary pen with the new temporary color
        CALL FGD_CREATE_TEMP_PEN(windowid, ipen, icolor, dfltthick)

        RETURN

        END

