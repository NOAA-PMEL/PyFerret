	SUBROUTINE PPLCMD(FROMI,LINEI,ISI,COMBUF,ICMDIM,ICMSZE)

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
* V54: *acm* 9/01 changes for dynamic allocation of PPLUS memory.
*                  This routine passes arguments through by a C "envelope" 
*                  routine pplcmd_c, which gets the global pointer to PPLUS 
*                  memory and passes it to pplcmd_f.F  which has the original code.
*                  Put the strings in common to avoid wierdness with strings in C.

c	SUBROUTINE PPLCMD(FROMI,LINEI,ISI,COMBUF,ICMDIM,ICMSZE)

      INCLUDE 'pplcmd_strings.cmn'

	CHARACTER FROMI*(*), LINEI*(*), COMBUF(ICMDIM)*(*)
        INTEGER   ICMDIM,ISI,ICMSZE     

c     internal parameter declarations:
      INTEGER      slen
      PARAMETER  ( slen = 255 )

      if (icmdim .gt. 5) 
     . print *, ' icmdim in pplcmd_strings.cmn too small', icmdim

      DO i = 1, ICMDIM
         PPLCOMBUF(i) = COMBUF(i)
      ENDDO

      PPLFROMI = FROMI
      PPLLINEI = LINEI

      CALL pplcmd_c  (ISI, ICMDIM, ICMSZE)

      RETURN
      END
