	LOGICAL FUNCTION UNIT_CHG_AX( cx, idim, trans )

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

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:   1/12/90
* V420: send back the transformation that has been applied

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* calling argument declarations
	INTEGER	cx, idim, trans

* internal variable declarations
* ...

* transformations are performed from x to t order.
* It is the LAST transformation that determines the units (at least so far)
* (probably a very imperfect test ... 11/95)
	DO 100 idim = 4, 1, -1
	   IF ( cx_trans(idim,cx) .EQ. trans_locate ) THEN
	      trans = punit_mod_loc
	      GOTO 200
	   ELSEIF (cx_trans(idim,cx) .EQ. trans_variance
     .	     .OR. ( cx_unstand_grid(cx)
     .	      .AND. (cx_regrid_trans(idim,cx).EQ.prgrd_variance
     .	        .OR. cx_regrid_trans(idim,cx).EQ.prgrd_modvar)  ) ) THEN
	      trans = punit_mod_variance
	      GOTO 200
	   ELSEIF (cx_trans(idim,cx) .EQ. trans_bad_pt
     .	      .OR. cx_trans(idim,cx) .EQ. trans_good_pt
     .	      .OR. ( cx_unstand_grid(cx)
     .	       .AND. (cx_regrid_trans(idim,cx).EQ.prgrd_good_pt
     .		 .OR. cx_regrid_trans(idim,cx).EQ.prgrd_modngd)  ) ) THEN
	      trans = punit_mod_npoints
	      GOTO 200
	   ENDIF
 100	CONTINUE

* no unit change
	UNIT_CHG_AX = .FALSE.
	RETURN
	
* yes, unit change
 200	UNIT_CHG_AX = .TRUE.
	RETURN

	END
