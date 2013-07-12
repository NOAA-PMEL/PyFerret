      SUBROUTINE DUP_COLOR_SET_OPACITY(windowid, icolor, opacity_frac)

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

*  For the active window, uses the RGB part of the given color ID and 
*  the opacity fraction to create a new temporary color.  The ID of 
*  this new color is returned in icolor.  Calls STOP if any problems.

        INCLUDE 'fgrdel.cmn'

* calling argument declarations:
        INTEGER windowid, icolor
        REAL*4 opacity_frac

* local argument declarations:
        INTEGER ier
        REAL*4 redfrac, grnfrac, bluefrac, opaqfrac

*       Sanity checks
        IF ( (windowid .LT. 1) .OR. 
     .       (windowid .GT. maxwindowobjs) ) THEN
            STOP 'DUP_COLOR_SET_OPACITY: Invalid windowid value'
        ENDIF
        IF ( windowobjs(windowid) .EQ. nullobj ) THEN
            STOP 'DUP_COLOR_SET_OPACITY: null windowobj'
        ENDIF
        IF ( (opacity_frac .LT. 0.0) .OR. (opacity_frac .GT. 1.0) )
     .     STOP 'Invalid opacity_frac passed to DUP_COLOR_SET_OPACITY'

*       Get the color fractions for the given color
        CALL FGD_GQCR(windowid, icolor, ier, 
     .                redfrac, grnfrac, bluefrac, opaqfrac)
        IF ( ier .NE. 0 )
     .     STOP 'Invalid icolor passed to DUP_COLOR_SET_OPACITY'

*       Create a new temporary color with the new opacity
        opaqfrac = opacity_frac
        CALL FGD_CREATE_TEMP_COLOR(windowid, icolor, 
     .                  redfrac, grnfrac, bluefrac, opaqfrac)

        RETURN

        END

