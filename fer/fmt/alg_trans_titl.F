	CHARACTER*20 FUNCTION ALG_TRANS_TITL( trans, cx )

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
