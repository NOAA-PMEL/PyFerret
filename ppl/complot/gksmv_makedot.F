        SUBROUTINE GKSMV_makedot
*
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

*  8/2001  acm  for PLOT/SYMBOL=DOT
*  Version of GKSMV to be called by make_dot.  Call GPM for polymarker
*  rather than GPL (polyline) 
*
*  10/2001 Buffer calls to GPM.  Rename X and Y to xdot, ydot for clarity.
*
* V73  3/18 *kms*  For clarity, append MD to makedot.cmn values IPT, XDOT, 
*                  and YDOT; make MDOT (99) a makedot.cmn parameter DOTSYMIM
* V73  3/18 *kms*  Clean up code for dots, which only need their coordinates
*                  and is where the marker is plotted. (Lines need pairs of 
*                  coordinates and are drawn between each pair of coordinates.)

        include 'pltcom_dat.decl'
        include 'PLTCOM.DAT'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'

        IF ( .NOT. GKSOPN )
     .      STOP 'GKSMV_makedot called when GKSOPN is .FALSE.'
        IF ( PEN ) THEN
*           Continue accumulating the coordinates for this marker
            CALL GKPLOT_makedot(A, B, 1)
        ELSE
*           Force plotting of the accumulated marker coordinates
*           and start a new accumulation of marker coordiantes
            CALL GKPLOT_makedot(A, B, 0)
        ENDIF
        RETURN
        END


        SUBROUTINE GKPLOT_makedot(XN, YN, IP)
        INCLUDE 'makedot.cmn'    ! 10/01 *acm*

        IF ( IP .EQ. 0 ) THEN
*           Force plotting of the accumulated marker coordinates
*           and start a new accumulation of marker coordiantes
            CALL MAKEDOTFLUSH
        ELSE
*           Add another coordinate for this marker 
            IF ( IPTMD .GE. MAXDOT ) THEN
*               Out of space - plot the accumulated marker coordinates
*               and start a new accumulation of marker coordiantes
                CALL MAKEDOTFLUSH
            ENDIF
        ENDIF
*       Add this marker coordinate
        IPTMD = IPTMD + 1
        XDOTMD(IPTMD) = XN
        YDOTMD(IPTMD) = YN
        RETURN
        END


        SUBROUTINE MAKEDOTFLUSH
        INCLUDE 'makedot.cmn'

        IF ( IPTMD .GT. 0 ) THEN
            CALL FGD_GPM(IPTMD, XDOTMD, YDOTMD)
            IPTMD = 0
        ENDIF
        RETURN
        END

