	LOGICAL FUNCTION TM_ITSA_MODEL_RUN(name, source, expnum)

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
* Determine if a particular data set represents a TMAP model run based on its
* name, only.

* Steve Hankin  NOAA/PMEL/TMAP

* 9/13/95 for Ferret version 4.2
* *kob* 10/96 - Linux port - changed function type from integer to
*                            logical
* Include files

* Argument definitions
	CHARACTER*(*) name, source, expnum

* Local definitions
	LOGICAL	TM_DIGIT
	INTEGER slen, ii, iiout
	CHARACTER*3 expdig

* initialize
	slen = LEN(name)

* name long enough to try decoding ("mtsa001" is minimal)?
	IF ( slen .LT. 7 ) GOTO 1000

* must start with "mt" or "mc"
	IF ( name(1:2).NE.'mt'
     . .AND. name(1:2).NE.'mc' ) GOTO 1000

* skip 1 character following "mt" and then look for a non-digit.
* (this kludge will work for "mtsa001" as well as "mt165w001" and "mt5n001")
	DO 100 ii = 4,slen
	  IF ( .NOT.TM_DIGIT(name(ii:ii)) ) GOTO 200
 100	CONTINUE
	GOTO 1000	! all digits!!

* there must be enough characters remaining for a 3-digit experiment code
 200	IF ( slen .LT. ii+3 ) GOTO 1000

* the next 3 characters must be digits
	iiout = 1
	DO 300 ii = ii+1,ii+3
	   IF ( TM_DIGIT(name(ii:ii)) ) THEN
	      expdig(iiout:iiout) = name(ii:ii)
	      iiout = iiout + 1
	   ELSE
	      GOTO 1000
	   ENDIF
 300	CONTINUE

* yes.  It's a TMAP MOM run
	source = 'MOM'
	expnum = '0'//expdig
	TM_ITSA_MODEL_RUN = .TRUE.
	RETURN

* nope.  It ain't a TMAP model run
 1000	TM_ITSA_MODEL_RUN = .FALSE.
	RETURN
 	END
