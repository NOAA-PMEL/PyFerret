	CHARACTER*(*) FUNCTION REPLACE_DEQ ( orig )

* replace "D=#" in variable titles with "D=name"

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  3/3/92
* V300:  bug fix: check if expanded length is within allowable limits
* V314: 8/22/94 *kob* IBM port - needed to add ifdef MANDATORY_FORMAT_WIDTH for
*				 I format descriptor
* V420: 10/5/95 - bug detected in syntax var2[d=1]- VAR1[d=2,G=[VAR2[D=1]]
*		- actual bug was that i1 is improperly set in branches to 10

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
        include 'tmap_format/tmap_dims.parm'
        include 'tmap_format/xdset_info.cmn_text'
        external xdset_info_data
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
#endif

* calling argument declarations
	CHARACTER*(*) orig

* internal variable declarations
	INTEGER TM_LENSTR1, i, i0, i1, i2, i3, eqpos, inlen, outlen,
     .       dset, nlen, maxlen

        CHARACTER*1 tab
        PARAMETER     ( tab = CHAR(9) )

* initialize
        i0 = 1
!	i1 = 1
        eqpos = 0
	inlen = LEN(orig)
        outlen = 0
	maxlen = LEN( REPLACE_DEQ )    ! 3/93
        REPLACE_DEQ = ' '

* search for next D=
 10     i1 = eqpos + 1			! 10/5: was "i1 = i1 + eqpos"
 11     eqpos = INDEX(orig(i1:), '=')
        IF ( eqpos .EQ. 0 ) GOTO 500
        eqpos = eqpos + i1 - 1
        DO 100 i = eqpos-1, i1, -1
           IF ( orig(i:i) .NE.' '
     .    .AND. orig(i:i) .NE.tab ) GOTO 110
 100    CONTINUE
        GOTO 500   ! blank is final character ??

* is it a "d" ?
 110    IF (orig(i:i).NE.'D' .AND. orig(i:i).NE.'d') GOTO 10

* make sure it's a "D" by itself instead of the end of another word
       IF ( orig(i-1:i-1).NE.' '
     . .AND. orig(i-1:i-1).NE.tab
     . .AND. orig(i-1:i-1).NE.'['
     . .AND. orig(i-1:i-1).NE.','
     . .AND. orig(i-1:i-1).NE.'/' ) GOTO 10

* got a "D=".  Is it followed by a number ?
        DO 200 i2 = eqpos+1, inlen
           IF ( orig(i2:i2) .NE.' '
     .    .AND. orig(i2:i2) .NE.tab ) GOTO 210
 200    CONTINUE
        GOTO 500   ! "=" is final character ??

* if it's "D=name" instead of "D=#" then ignore it
 210    IF ( orig(i2:i2).LT."1" .OR. orig(i2:i2).GT."9" ) GOTO 10

* get the data set number
        DO 300 i3 = i2+1, inlen
           IF ( orig(i3:i3).LT."0" .OR. orig(i3:i3).GT."9" ) GOTO 310
 300    CONTINUE
        GOTO 500   ! digit is final character ??
 310    i3 = i3 - 1
#ifdef MANDATORY_FORMAT_WIDTHS
        READ (orig(i2:i3),*,ERR=5000) dset
#else
        READ (orig(i2:i3),'(I)',ERR=5000) dset
#endif
        IF (dset.LT.1 .OR. dset.GT.maxdsets) GOTO 10    ! error if so

* replace D=# with D=name
        nlen = TM_LENSTR1(ds_name(dset))
        IF ( outlen .EQ. 0 ) THEN
           REPLACE_DEQ = orig(:eqpos)//ds_name(dset)(:nlen)
        ELSE
           REPLACE_DEQ = REPLACE_DEQ(:outlen)
     .               //orig(i0:eqpos)//ds_name(dset)(:nlen)
        ENDIF
        outlen = MIN( maxlen, outlen+eqpos-i0+1+nlen )   ! 3/93
        i1 = i3 + 1  ! skip over the dset # characters
        i0 = i1
        GOTO 11 

* tag on whatever is left over
 500    IF ( outlen .EQ. 0 ) THEN
           REPLACE_DEQ = orig
        ELSE
           REPLACE_DEQ = REPLACE_DEQ(:outlen)//orig(i0:inlen)
        ENDIF

 5000   RETURN
	END
