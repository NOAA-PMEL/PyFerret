
        SUBROUTINE RESET_LABSIZ (ywid, ywidt, abssz, noylab)

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
* 11/2011 ACM Fixing ticket 1896:
*
* Use origin and label-size info that is in common to possibly
* reset the axis-label size for a vertical axis, so the labels dont
* run off the edge.

* 5/12  Fix the way the size is rescaled.
*       Issue a note when the label size is reset.
* V686 *acm* 11/13 Allow symbol names to be up to 120 characters long

	REAL ywid, ywidt, abssz
	LOGICAL noylab

        include 'axis_inc.decl'
        include 'AXIS.INC'
        include 'labcom_inc.decl'
        include 'LABCOM.INC'

	REAL xl, frac, dhylab
	CHARACTER*100 note
	CHARACTER*120 argsym
	INTEGER i1, i2, len

* This is irrelevant if there is no margin anyway (viewport defined with /AXES
* up against the edge of the page.)
	IF (xorg .LT. 1.e-6) RETURN

        dhylab = 2.5*hylab
	IF (noylab) dhylab = 0.
        xl = xorg-(ywid+ywidt+dhylab)
        IF (xl.LT.0) THEN
	   frac = xorg/(ywid+ywidt+dhylab)

	   IF (frac .LT. 0) frac = 0.5  ! margin is too small for this method. Punt.
	   WRITE (note, 100) abssz, abssz* frac
	   CALL WARN (note)
	   abssz = abssz* frac

	i1 = 1
	i2 = 1
 1710   CALL LSTSYM(argsym,note,len,i1,i2)	! get next symbol
	IF(i2.EQ.0)THEN
           IF(argsym(:12) .EQ. 'LAST_GO_FILE')THEN
	      CALL WARN (note)
	   ELSE
              GOTO 1710
	   ENDIF
	      CALL LSTSYM(argsym,note,len,i1,i2) 
	   ENDIF
	ENDIF

 100	FORMAT ('Adjusting Y-axis label size from ', f5.3, ' to ', f5.3, 
     .          ', to avoid running off page. ')
	RETURN
	END
