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
*DECK CMESH
      SUBROUTINE CMESH (DX,APB,AMB,WID,N,NDIR)
C
C  COSINE MESH
C  APB=A+B = COARSE RESOLUTION
C  AMB=A-B=FINE RESOLUTION
C  WID = WIDTH BETWEEN COARSE & FINE RESOLUTION
C  N = NUMBER OF VARIABLE SPACES CALCULATED TO SPAN WID
C  NDIR = (1,-1) = SPACING (INCREASES,DECREASES) WITH INCREASING INDEX
C
      REAL DX(160)	! changed 100 to 160 : sh 6-13-85
	REAL	apb, amb, wid, a, b, c, pi, rn
	INTEGER	n, ndir, i, ii
      DATA PI/3.14159/
      A=.5*(APB+AMB)
      B=.5*(APB-AMB)
      C=WID/A
      N=C+.1
      RN=1./FLOAT(N)
      DO 20 II=1,N
      IF (NDIR.EQ.1) I=N+1-II
      IF (NDIR.EQ.-1)  I=II
   20 DX(I)=A+B*COS(PI*RN*(II-.5))
      RETURN
      END
