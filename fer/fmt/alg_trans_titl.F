	CHARACTER*20 FUNCTION ALG_TRANS_TITL( trans, cx )

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
* return the title of the given transformation

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
*
* revision 0.0 - 4/28/87
* V430: 7/96 if a 4D transformation then pre-pend the axes involved

* calling argument declarations:
	INTEGER	trans, cx

* internal variable declarations:
	INTEGER	i, i4d, ndim, tcode
	CHARACTER prefix*5

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xtext_info.cmn'
	include 'ferret_cmn/xalgebra.cmn'
	include 'ferret_cmn/xcontext.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XTEXT_INFO.CMN'
	INCLUDE 'FERRET_CMN:XALGEBRA.CMN'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
#endif

* identify the code from known codes
	DO 100 tcode = 1, num_transforms
	   IF ( trans .EQ. alg_trans_num(tcode) )	GOTO 200
 100	CONTINUE

* unknown code
	ALG_TRANS_TITL = '???'
	RETURN

* is this a 4D transformation? If so, include axis prefix to the transformation
 200	DO 250 i4d = 1, num_4d_trans
	   IF ( trans .EQ. alg_4d_trans(i4d) ) THEN
	      prefix = ' '
	      ndim = 0
	      DO 240 i = 1, 4
	         IF (cx_trans(i,cx) .EQ. trans) THEN
	            ndim = ndim + 1
	            prefix = prefix(:ndim)//ww_dim_name(i)
	         ENDIF
 240	      CONTINUE
	      ALG_TRANS_TITL = prefix(2:ndim+1)//alg_trans_title(tcode)(3:)
	      RETURN
	   ENDIF
 250	CONTINUE	

* not a 4D transform -- simply return the text
	ALG_TRANS_TITL = alg_trans_title(tcode)
	RETURN
	END
