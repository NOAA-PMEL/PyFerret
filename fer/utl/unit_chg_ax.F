	LOGICAL FUNCTION UNIT_CHG_AX( cx, idim, trans )

* if a transformation has changed the units of this variable return the info

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
