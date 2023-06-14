	SUBROUTINE GKSMV
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
*
*
*	Mod 1.9.90 J Davison to reduce the maximum #pts in any one call to 
*	polyline to 100 as GKS 4.0 under VMS 5.2, i.e., DECWindows, blows up
*	with too many points
*
C
C	CALL GKSMV
C
C	This routine creates a binary file that can be used to
c	construct device dependent plotting code
c
c v6.61 ACM 4/2010 Fix bug 1396 - dots between dashes on some displays.
c                  Do not make a call to GPL on "move with the pen up".      
C
C v6.93 KMS 11/2014 Rename IPT, X, and Y in GKPLOT to ipline, xpline, 
C                   and ypline, and move to a common block so that it 
C                   does not depend on local variables being static 
C                   (similar to what was done in gksmv_makedot.F).  
C                   Increase the maximum back up to 10000.

	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'gkscm1_inc.decl'
	include 'GKSCM1.INC'

        IF ( .NOT. GKSOPN )
     .      STOP 'GKSMV called when GKSOPN is .FALSE.'

	IF(PEN)THEN
	   CALL GKPLOT(A,B,1)
	ELSE
	   CALL GKPLOT(A,B,0)
	ENDIF
	RETURN
	END

*jd*	SUBROUTINE GKPLOT(XN,YN,IP,II) Number of args reduced to 3 for MAC 7.92
	SUBROUTINE GKPLOT(XN,YN,IP)

        INCLUDE 'gkplot.cmn'
        DATA ipline/0/

        IF ( IP .EQ. 0 ) THEN
*           Draw the accumulated connected lines
*           and start a new accumulation of lines
            IF ( ipline .GT. 0 ) THEN
                CALL DOIT_gksmv(ipline,xpline,ypline)
            ENDIF
            ipline=1
        ELSE
*           Add to the accumulated connected lines
            IF ( ipline .GE. maxpline ) THEN
*              No space left - draw the lines and 
*              start a new accumulation, continuing
*              from the last point of the last line
               CALL DOIT_gksmv(ipline,xpline,ypline)
               xpline(1) = xpline(ipline)
               ypline(1) = ypline(ipline)
               ipline=2
            ELSE
               ipline=ipline+1
            ENDIF
        ENDIF
        xpline(ipline) = XN
        ypline(ipline) = YN
	RETURN
	END

        SUBROUTINE PPLGFLUSH
        INCLUDE 'gkplot.cmn'
        CALL DOIT_gksmv(ipline,xpline,ypline)
        ipline=0
        RETURN
        END

	SUBROUTINE DOIT_gksmv(IPT,X,Y)
	DIMENSION X(*),Y(*)
	IF(IPT.LE.0)RETURN
	IF(IPT.EQ.1)THEN
*           Singleton - draw as a zero length line
	    X(2)=X(1)
	    Y(2)=Y(1)
	    IPT=2
	ENDIF
	CALL FGD_GPL(IPT,X,Y)
	RETURN
	END
