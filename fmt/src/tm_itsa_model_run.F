	LOGICAL FUNCTION TM_ITSA_MODEL_RUN(name, source, expnum)

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
