      SUBROUTINE getlabfonts (lab, lnbeg, lnend, jln, npnew, nfnew, 
     .           newfont, newpen, nchr)

* Argument declarations

      CHARACTER*(*) lab, newfont, newpen
      INTEGER lnbeg(*), lnend(*), npnew, nfnew, nchr, jln

* Local variable definitions

      CHARACTER*1 fcode
      INTEGER TM_LOC_STRING, ioc, loc

      npnew = 0
      nfnew = 0
      newfont = ""
      newpen = ""

      nchr = lnend(jln) - lnbeg(jln) + 1
      ioc = 0
      loc = TM_LOC_STRING(lab(lnbeg(jln):lnend(jln)),"@",ioc)

      IF ( loc .GT. 0) THEN
         fcode = lab(lnbeg(jln)+loc:lnbeg(jln)+loc)
         IF (fcode .EQ. 'P'  .OR.  fcode .EQ. 'p') THEN
            newpen(1:3) = lab(lnbeg(jln)+loc-1:lnbeg(jln)+loc+1)
            npnew = 3
         ELSE
            newfont(1:3) = lab(lnbeg(jln)+loc-1:lnbeg(jln)+loc+1)
            nfnew = 3

         ENDIF
         nloc = loc

! Save the last pen and font change on this line.

         DO WHILE (nloc .NE. 0) 
            ioc = ioc + 1
            nloc = TM_LOC_STRING(lab(lnbeg(jln):lnend(jln)),"@",ioc)
            IF (nloc .GT. 0) THEN
               fcode = lab(lnbeg(jln)+nloc:lnbeg(jln)+nloc)
               IF (fcode .EQ. 'P'  .OR.  fcode .EQ. 'p') THEN
                 newpen(1:3) = lab(lnbeg(jln)+nloc-1:lnbeg(jln)+nloc+1)
                 npnew = 3
               ELSE
                 newfont(1:3) = lab(lnbeg(jln)+nloc-1:lnbeg(jln)+nloc+1)
                 nfnew = 3
               ENDIF
            ENDIF
         ENDDO
      ENDIF

      RETURN
      END
