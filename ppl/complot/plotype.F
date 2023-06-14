      SUBROUTINE PLTYPE(ICODE)
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
*
*	Modified 3.15.89 J Davison JISAO/PMEL/TMAP
*	For GKS plots (pltype = 3 or 4) captures decision to write to
*	GKS metafile, and sets up GKS as required.
*  11/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals. In PyFerret pltype is 3
*
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     PLTTER DEPENDENT ROUTINE
C     THIS ROUTINE MUST BE MODIFIED EVERY TIME NEW PLT DEVICES ARE ADDE
C     OR OLD ONES ARE DELETED.
C
C
C     CALL PLTYPE(ICODE)
C
C     THIS SUBROUTINE SELECTS THE DEVICE OR DEVICES THAT WILL BE
C     USED FOR PLTTING.
C
C     THE VARIOUS COMBINATIONS OF PLTTING DEVICES THAT CAN BE
C     DRIVEN BY COMPLT AND THE ASSOCIATED PLT TYPES ARE LISTED
C     BELOW:
C	 GKS AND ZETA PLOTTER               -    4
C        GKS ONLY                           -    3
C        TEKTERMINAL AND ZETA PLOTTER       -    2
C        TEKTERMINAL ONLY                   -    1
C        ZETA PLOTTER ONLY                  -    0
C	 HP7475 PLOTTER ONLY		    -   -1
C	 HP7475 PLOTTER AND TEKTERMINAL     -   -2
C
C
	integer	icode

	include 'pltcom_dat.decl'
	include 'PLTCOM.DAT'
	include 'cmrdl_inc.decl'
	include 'CMRDL.INC'

	IF(PLTFLG)THEN
	    PEN=.FALSE.
	    PLTFLG=.FALSE.
	ENDIF
      PTYPE = ICODE

*     DETERMINE IF METAFILE WILL BE USED IN GKS, OPEN GKS IF NOT OPEN
	if ((ptype .eq. 3) .and. icom(1:6) .ne. 'CLSPLT') 
     .		call set_gks_metafile

C     REINITIALIZE THE TEKTERMINAL TYPE IN CASE THIS IS NOT THE
C     FIRST TIME THIS SUBROUTINE HAS BEEN CALLED.

      CALL TKTYPE(4010)

      RETURN
C
      END
