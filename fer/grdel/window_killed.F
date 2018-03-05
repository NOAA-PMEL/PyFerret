      SUBROUTINE WINDOW_KILLED( deadwinobj )

*
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
*
* Adjusts the ferret engine as if the 'cancel window' had been issued,
* but assumes that window has already been killed; e.g., from using the
* window frame 'X' button.

        IMPLICIT NONE

        include 'tmap_dims.parm'
        include 'xplot_state.cmn'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'
        include 'fgrdel.cmn'

        REAL*8 deadwinobj
        INTEGER k, windowid, success

*       ignore if not a valid window object
        IF ( deadwinobj .EQ. nullobj ) RETURN
*       find the number of this window object in Ferret's list
        DO k = 1, maxwindowobjs
            IF ( windowobjs(k) .EQ. deadwinobj ) THEN
                windowid = k
                GOTO 10
            ENDIF
        ENDDO
*       window object not found - ignore this call
        RETURN
*       window object found
 10     CONTINUE

*       if this window was already closed, ignore this call
        IF ( .NOT. wn_open(windowid) ) RETURN

*       mark the window as closed since it is already killed
        wn_open(windowid) = .FALSE.

*       set a new active window if the dead window had been the active window
        IF ( wn_active(windowid) ) THEN
           wn_active(windowid) = .FALSE.
           DO k = 1, max_windows
              IF ( wn_open(k) ) THEN
                 CALL FGD_GACWK(k)
                 wn_active(k) = .TRUE.
                 wsid = k
                 GOTO 20
              ENDIF
           ENDDO
*             no active window
           wsid = 0
        ENDIF

*       close the specified window, ignoring errors
 20     CONTINUE

*       Delete any remaining objects associated with this window.
*       Clear values associated with those objects just to be safe.
        DO k = 1, maxsymbolobjs
            IF ( symbolobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDSYMBOLDEL(success, symbolobjs(k, windowid))
                symbolobjs(k, windowid) = nullobj
            ENDIF
        ENDDO
        DO k = 1, maxpenobjs
            IF ( penobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDPENDEL(success, penobjs(k, windowid))
                penobjs(k, windowid) = nullobj
            ENDIF
            pencolor(k, windowid) = 0
            penwidth(k, windowid) = -1.0
        ENDDO
        DO k = 1, maxfontobjs
            IF ( fontobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDFONTDEL(success, fontobjs(k, windowid))
                fontobjs(k, windowid) = nullobj
            ENDIF
        ENDDO
        DO k = 1, maxbrushobjs
            IF ( brushobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDBRUSHDEL(success, brushobjs(k, windowid))
                brushobjs(k, windowid) = nullobj
            ENDIF
            brushcolor(k, windowid) = 0
        ENDDO
        DO k = 1, maxcolorobjs
            IF ( colorobjs(k, windowid) .NE. nullobj ) THEN
                CALL FGDCOLORDEL(success, colorobjs(k, windowid))
                colorobjs(k, windowid) = nullobj
            ENDIF
            colorrgba(1, k, windowid) = -1.0
            colorrgba(2, k, windowid) = -1.0
            colorrgba(3, k, windowid) = -1.0
            colorrgba(4, k, windowid) = -1.0
        ENDDO

*       Clear the DPI values just to be safe
        windowdpix(windowid) = 0.0
        windowdpiy(windowid) = 0.0

*       Remove this window object from Ferret's list of objects.
*       Assumes bindings will be dealt with by the caller.
        windowobjs(windowid) = nullobj

        RETURN
        END
