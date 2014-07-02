      SUBROUTINE getlabfonts (lab, lnbeg, lnend, jln, fontprefix,
     .          npnew, nfnew, newfont, newpen, nchr)
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
* v580 10/05 *acm* Fix for bug1003; pen may be changed by @Cnnn as well
*                  as @Pn. In saving info about color/font changes from
*                  one line to the next in multi-line lables, allow for
*                  @Cnnn syntax.
* V683 12/12 *acm* Fix for applying a font setting in multiple line labels
* V69+  5/14 *acm* Fix for ascii font in multi-line labels, ticket 2176

* Argument declarations

      IMPLICIT NONE
      CHARACTER*(*) lab, newfont, newpen, fontprefix
      INTEGER lnbeg(*), lnend(*), npnew, nfnew, nchr, jln

* Local variable definitions

      CHARACTER*1 fcode, fnext, ichr
      INTEGER TM_LOC_STRING, ioc, loc, ii, nloc, eloc
      LOGICAL digit, isc, ascfnt

      npnew = 0
      nfnew = 0
      newfont = ""
      newpen = ""

* Is the current setting an ASCII font
      ascfnt = (fontprefix(1:2).EQ.'@A' .OR. fontprefix(1:2).EQ.'@a') 

      nchr = lnend(jln) - lnbeg(jln) + 1
      ioc = 0
      loc = TM_LOC_STRING(lab(lnbeg(jln):lnend(jln)),"@",ioc)

      IF ( loc .GT. 0) THEN

* Is the ASCII font escaped before the first font/pen setting?
         DO ii = lnbeg(jln), lnbeg(jln)+loc
            ichr = lab(ii:ii)
            IF (ichar(ichr) .eq.27 .AND. ii .LT. lnbeg(jln)+loc+1) THEN
               ascfnt = .FALSE.
	       eloc = ii
	       GOTO 100
	    ENDIF
         ENDDO
 100     CONTINUE

* Font/pen settings are only applied if it's not the ascii font or
* the @ was escaped.

	 IF (.NOT. ascfnt) THEN
            fcode = lab(lnbeg(jln)+loc:lnbeg(jln)+loc)
	    fnext = lab(lnbeg(jln)+loc+1:lnbeg(jln)+loc+1)
   
	    digit = (fcode .EQ. 'C'  .OR.  fcode .EQ. 'c') 
	    digit = ( fnext .GE. '0' .AND. fnext .LE. '9')
	    isc = ((fcode .EQ. 'C'  .OR.  fcode .EQ. 'c') .AND.digit) 

            IF (fcode .EQ. 'P'  .OR.  fcode .EQ. 'p') THEN
               newpen(1:3) = lab(lnbeg(jln)+loc-1:lnbeg(jln)+loc+1)
               npnew = 3
            ELSE IF ((fcode .EQ. 'C'  .OR.  fcode .EQ. 'c') .AND.digit) THEN
               newpen(1:5) = lab(lnbeg(jln)+loc-1:lnbeg(jln)+loc+3)
               npnew = 5
            ELSE 
               newfont(1:3) = lab(lnbeg(jln)+loc-1:lnbeg(jln)+loc+1)
               nfnew = 3

            ENDIF
	 ENDIF
         nloc = loc

* Is the current setting an ASCII font
         ascfnt = (fontprefix(1:2).EQ.'@A' .OR. fontprefix(1:2).EQ.'@a') 

! Save the last pen and font change on this line.

         DO WHILE (nloc .NE. 0) 
            ioc = ioc + 1
            nloc = TM_LOC_STRING(lab(lnbeg(jln):lnend(jln)),"@",ioc)
            IF (nloc .GT. 0) THEN

* Is the ASCII font escaped before the first font/pen setting?
               DO ii = eloc+1, lnbeg(jln)+nloc+1
                  ichr = lab(ii:ii)
                  IF (ichar(ichr) .eq.27 .AND. ii .LT. lnbeg(jln)+nloc+1)  THEN
                     ascfnt = .FALSE.
	             GOTO 200
                  ENDIF
               ENDDO
 200           CONTINUE

* Font/pen settings are only applied if it's not the ascii font or
* the @ was escaped.

	       IF (.NOT. ascfnt) THEN
                  fcode = lab(lnbeg(jln)+nloc:lnbeg(jln)+nloc)
                  IF (fcode .EQ. 'P'  .OR.  fcode .EQ. 'p') THEN
                    newpen(1:3) = lab(lnbeg(jln)+nloc-1:lnbeg(jln)+nloc+1)
                    npnew = 3
                  ELSE IF (fcode .EQ. 'C'  .OR.  fcode .EQ. 'c') THEN
                    newpen(1:5) = lab(lnbeg(jln)+nloc-1:lnbeg(jln)+nloc+3)
                    npnew = 5
                  ELSE

                    newfont(1:3) = lab(lnbeg(jln)+nloc-1:lnbeg(jln)+nloc+1)
                    nfnew = 3
                  ENDIF

	       ENDIF ! .not.ascfnt

            ENDIF ! nloc.gt.0
	    	       
         ENDDO ! while nloc eq 0

      ENDIF    ! loc gt 0

      RETURN
      END
