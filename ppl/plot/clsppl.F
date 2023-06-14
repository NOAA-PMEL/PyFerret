        SUBROUTINE CLSPPL

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
*       Modified 9.27.88 J Davison TMAP/JISAO/PMEL
*       to close GKS properly if it is opened.
*       J Davison 11.28.88 modified for compatibility to v1.1 PPL
*       k obrien 3.11.97 added ifdef for linux port to ignore "unlink"
* V602: *acm* 1/07 Changes for metafile-batch mode
*       *kms* 10/11 set GKSOPN and GKS_OPEN flags to .FALSE.
*                   after closing GKS
*  12/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals

        include 'symkey_inc.decl'
        include 'SYMKEY.INC'
        include 'cmdcom_inc.decl'
        include 'CMDCOM.INC'
        include 'lunits_inc.decl'
        include 'LUNITS.INC'
        include 'pltcom_dat.decl'
        include 'PLTCOM.DAT'
        include 'gkscm2.cmn'

C
C       CLOSES FILES USED BY PLOT5+
C
	
        CALL ATFLSH
        call dbmclose(keyln1)
        call close_gks !See bottom of this file. J Davison 9.27.88
 
        GKSOPN = .FALSE.
        GKS_OPEN = .FALSE.
        RETURN
        END

        SUBROUTINE CLOSE_GKS

*       Written to close GKS if it is open, deactivate and close workstations
*       J Davison 9.27.88

        include 'pltcom_dat.decl'
        include 'PLTCOM.DAT'
        include 'gkscm2.cmn'

        integer member,er_status,num_active,rmember,i,j,
     .          num_open, istate

********************************************************************************


        CALL FGD_GQOPS(istate)  ! In batch mode the graphics may not be active now. 

        if (gksopn) then 
           if ( istate .GE. 2 ) then

              member = 1
              call FGD_GQACWK(member,er_status,num_active,rmember)
              if (num_active .ge. 1) then

                 do i = 1,num_active
                    call FGD_GQACWK(1,er_status,j,rmember)
                    call FGD_GDAWK(rmember)
	         enddo

	      endif

              call FGD_GQOPWK(member,er_status,num_open,rmember)
              if (num_open .lt. 1) goto 40

              do i = 1,num_open
                 call FGD_GQOPWK(1,er_status,j,rmember)
                 call FGD_GCLWK(rmember)
	      enddo

           endif    ! istate
 40        call FGD_GCLKS()
        end if

        return
        end
