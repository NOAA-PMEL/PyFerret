	CHARACTER*(*) FUNCTION TM_CLEAN_FILENAME ( pathname )

* remove the directory path and the extension from a full pathname leaving
* only the filename

*  written 6/92 by S.H. for PMEL/TMAP
* 5/93 - *sh* modified so endings other than .cdf and .des are returned

* argument
	CHARACTER*(*) pathname

* local variable
        INTEGER idot, islash, plen

* initialize
        plen = LEN( pathname )

* locate the last dot in the name
        DO 100 idot = plen, 1, -1
 100    IF ( pathname(idot:idot) .EQ. '.' ) GOTO 190
        idot = plen + 1    ! no dot
 190    idot = idot - 1

* 5/93 other than .des or .cdf should not be cleaned off 
	IF ( plen-idot .EQ. 4 ) THEN
	   IF (   pathname(idot+1:plen) .NE. '.cdf'
     .      .AND. pathname(idot+1:plen) .NE. '.des' ) idot = plen
	ELSE
	   idot = plen
	ENDIF

* locate the last slash in the name
        DO 200 islash = plen, 1, -1
 200    IF ( pathname(islash:islash) .EQ. '/' ) GOTO 290
        islash = 0    ! no slash
 290    islash = islash + 1

* check for pathological name
        IF ( islash .GT. idot ) idot = plen

        TM_CLEAN_FILENAME = pathname(islash:idot)

	RETURN
	END
