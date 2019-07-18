	SUBROUTINE CLRKLAB (labstring)

	INCLUDE 'parampl5_dat.decl'
	INCLUDE 'PARAMPL5.DAT'
	INCLUDE 'cont_inc.decl'
	INCLUDE 'CONT.INC'
	include		'cmrd_inc.decl'
	include		'CMRD.INC'   

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

C 
C   Store strings to be used in colorbar labels, from label which is
C   the whole set of strings, up to 20 of them, separated by ampersands.
C
* V7.4+ 8/2018 ACM for DSG-enabled Ferret
* V7.51 6/2019 ACM Use Ampersand to separate strings, else the pplus parser is confused

	CHARACTER*(*) labstring

	INTEGER TM_LENSTR, i, slen, istr, ilab 
	CHARACTER*1 c, amper
	CHARACTER*20 TM_FMT_SNGL

C Initialize

	DO i = 1, 20
	   colorkey_labs(i) = " "
	ENDDO

	slen = TM_LENSTR(labstring)

	IF ( slen .EQ. 0 )THEN

* Numeric values were split up and put into P(). Turn them into strings.
	   IF (M.gt.0) THEN
	      DO i = 1, M
	         colorkey_labs(i) = TM_FMT_SNGL (P(i),  15, 20, slen)
	      ENDDO
	   ENDIF
	   GOTO 10

	ENDIF

* Strings: split up the label-string at the ampersands.

	amper = "&"
	istr = 1
	ilab = 1

	DO i = 1, slen
	   c = labstring(i:i)
	   IF (c .EQ. amper) THEN
	      ilab = ilab+1
	      istr = 1
	   ELSE
	      colorkey_labs(ilab)(istr:istr) = c
	      istr = istr+1
	   ENDIF
	ENDDO

10	CONTINUE


	RETURN
	END
