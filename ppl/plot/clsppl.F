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
