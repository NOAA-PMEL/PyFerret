	SUBROUTINE CLSPPL

*	Modified 9.27.88 J Davison TMAP/JISAO/PMEL
*	to close GKS properly if it is opened.
*	J Davison 11.28.88 modified for compatibility to v1.1 PPL
*	k obrien 3.11.97 added ifdef for linux port to ignore "unlink"
#ifdef unix
	include 'tmap_pplv11inc/symkey_inc.decl'
	include 'pplv11inc/SYMKEY.INC'
	include 'tmap_pplv11inc/cmdcom_inc.decl'
	include 'pplv11inc/CMDCOM.INC'
	include 'tmap_pplv11inc/lunits_inc.decl'
	include 'pplv11inc/LUNITS.INC'
	include 'tmap_pplv11inc/pltcom_dat.decl'
	include 'pplv11inc/PLTCOM.DAT'
#else
	INCLUDE 'TMAP_PPLV11INC:SYMKEY_INC.DECL'
	INCLUDE 'PPLV11INC:SYMKEY.INC'
	INCLUDE 'TMAP_PPLV11INC:CMDCOM_INC.DECL'
	INCLUDE 'PPLV11INC:CMDCOM.INC'
	INCLUDE 'TMAP_PPLV11INC:LUNITS_INC.DECL'
	INCLUDE 'PPLV11INC:LUNITS.INC'
	INCLUDE 'TMAP_PPLV11INC:PLTCOM_DAT.DECL'
	INCLUDE 'PPLV11INC:PLTCOM.DAT'
#endif
#ifdef core
        include 'pltinc/CORECM.INC'
#endif
C
C	CLOSES FILES USED BY PLOT5+
C
	CALL ATFLSH
#ifdef unix
        call dbmclose(keyln1)
#ifndef F90_NO_UNLINK
        call unlink('PPL$KEY.001.pag')
        call unlink('PPL$KEY.001.dir')
#endif
        call close_gks !See bottom of this file. J Davison 9.27.88
#else
        call close_gks !See bottom of this file. J Davison 9.27.88
	CLOSE(KEYLN1,DISP='DELETE')
#endif
#ifdef core
        if(gksopn)then
            call closeretainseg()
            call deselectvwsurf(vsurf)
            call terminatecore()
        endif
#endif
 
	IF(PLTFLG)THEN
	    PEN = .FALSE.
	    CALL ZABMV
	    CALL BINFSH
	    PLTFLG = .FALSE.
	ENDIF
	RETURN
	END

        SUBROUTINE CLOSE_GKS

*       Written to close GKS if it is open, deactivate and close workstations
*       J Davison 9.27.88

#ifdef unix
        include         'tmap_pplv11inc/pltcom_dat.decl'
        include         'pplv11inc/PLTCOM.DAT'
#else
        INCLUDE         'PPLV11INC:pltcom.dat'
#endif

        integer         member,er_status,num_active,rmember,i,j,num_open

********************************************************************************

        if (gksopn) then
	   member = 1
           call gqacwk (member,er_status,num_active,rmember)
           if (num_active .lt. 1) goto 20

           do 10 i = 1,num_active
              call gqacwk (1,er_status,j,rmember)
              call gdawk (rmember)
10         continue

20         call gqopwk (member,er_status,num_open,rmember)
           if (num_open .lt. 1) goto 40

           do 30 i = 1,num_open
              call gqopwk (1,er_status,j,rmember)
              call gclwk (rmember)
30         continue

40         call gclks ()
        end if

        return
        end
