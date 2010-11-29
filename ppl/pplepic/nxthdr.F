      SUBROUTINE NXTHDR(ITAPEI)
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
C**    @(#)nxthdr.f	1.1    3/10/88
C**
C**
C***********************************************************************        C**
C**                 EPIC
C**
C***********************************************************************        C**
C**
C  THIS SUBROUTINE WILL READ IN THE NEXT HEADER IN A MULTI HEADER 
C  FILE. IT WILL PLACE THE DATA IN ARRAY BUFFER.
C
C  PROGRAMMED 10/84 NY M.VERSCHELL
C
      CHARACTER*80 BUFFER(8)
      COMMON /comnxt/ BUFFER
C  
C  READ IN NEXT HEADER
C
      READ (ITAPEI,END=99) BUFFER(1)
      GOTO 100
   99 READ (ITAPEI,END=9000) BUFFER(1)
  100 DO 150 J=2,8
      READ (ITAPEI,END=9000) BUFFER(J)
  150 CONTINUE 
	RETURN
C
C  HERE BEGINS THE ERROR HANDLING CODE
C
 9000 WRITE (6,9100) NERROR
 9100 FORMAT(1X,'FORTRAN RUN TIME ERROR #',I2)
      IF (NERROR .EQ. -1) WRITE (6,9200)
 9200 FORMAT(1X,'END OF FILE READ BEFORE ALL OF HEADER READ')
      IF (NERROR .EQ. 29) WRITE (6,9300)
 9300 FORMAT(1X,'TAPE 1 NOT ASSIGNED (FOR001.DAT NOT FOUND)')
      IF (NERROR .EQ. 61) WRITE (6,9400)
 9400 FORMAT(1X,'ERROR IN CONVERSION OF NCAST OR NDATA')
      STOP
      END
