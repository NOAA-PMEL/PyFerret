      SUBROUTINE HDRT1 (ITAPEI, NDATA, DI, TI, DF, TF, DT)
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
C**    @(#)hdrt1.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
C
C  THIS SUBROUTINE WILL READ FIRST HEADER FROM AN EPIC TIME SERIES DATA FILE ON
C  TAPE ITAPEI. IT WILL RETURN INITIAL AND FINAL DATES (TI,TF) AND DT.
C  THE WHOLE HEADER WIL BE RETURNED IN THEADER.
C
C  PROGRAMMED 10/84 BY M. VERSCHELL
C
C  MODIFIED 3/86, NNS.
C
      CHARACTER*80 THEADER(8)
      COMMON /COMTHDR/ THEADER
C
C  READ IN HEADER FROM TAPE ITAPEI
C
      READ (ITAPEI) THEADER(1)
      GOTO 100
  100 DO 150 J=2,8
      READ (ITAPEI) THEADER(J)
  150 CONTINUE 
C
C  IF HEADER READ IN CORRECTLY, ISOLATE AND CONVERT STARTING TIME 
C
      READ(THEADER(1)(55:78),200) DI,TI,DF,TF
  200 FORMAT(F6.0,X,F4.0,2X,F6.0,X,F4.0)
      READ(THEADER(3)(55:78),300) NDATA,DT
  300 FORMAT(I6,E18.0)
      RETURN
      END
