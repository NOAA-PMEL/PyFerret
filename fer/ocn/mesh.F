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
      SUBROUTINE MESH (DX,DY)
      INTEGER  IM, JM, NX, NY
      PARAMETER (IM=160,JM=100,NX= 1,NY=3)
	INTEGER l, m, n
      REAL DX(IM),DY(JM)
      REAL XMAX(NX),XMIN(NX),XWID(NX)
      REAL YMAX(NY),YMIN(NY),YWID(NY)
	INTEGER	ID(NX), JD(NY)
      DATA XMAX/1.0/
      DATA XMIN/1.0/
      DATA XWID/160.0/
      DATA YMAX/2.3333334,.3333334,2.8666667/
      DATA YMIN/3*.3333334/
      DATA YWID/20.,20.,40./
      DATA ID/1/
      DATA JD/-1,2*1/
      M=1
      DO 5 L=1,NX
      CALL CMESH (DX(M),XMAX(L),XMIN(L),XWID(L),N,ID(L))
      M=M+N
    5 CONTINUE
      M=1
      DO 8 L=1,NY
      CALL CMESH (DY(M),YMAX(L),YMIN(L),YWID(L),N,JD(L))
      M=M+N
    8 CONTINUE
      RETURN
      END
