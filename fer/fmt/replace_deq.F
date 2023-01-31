	CHARACTER*(*) FUNCTION REPLACE_DEQ ( orig )

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
* replace the number found in a [D=#]" syntax with the dataset name

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V230:  3/3/92
* V300:  bug fix: check if expanded length is within allowable limits
* V314: 8/22/94 *kob* IBM port - needed to add ifdef MANDATORY_FORMAT_WIDTH for
*				 I format descriptor
* V420: 10/5/95 - bug detected in syntax var2[d=1]- VAR1[d=2,G=[VAR2[D=1]]
*		- actual bug was that i1 is improperly set in branches to 10
* 2/03 *kob* - g77 port - g77 won't allow intrinsic functions in PARAMETER
*                         statements.  use an octal constant instead
*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)
* 11/16 *sh* support for aggregation.member syntax

        IMPLICIT NONE
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include	'xvariables.cmn'
        include 'xdset_info.cmn_text'
        external xdset_info_data

* calling argument declarations
	CHARACTER*(*) orig

* internal variable declarations
	INTEGER TM_LENSTR1, FIND_DSET_NUMBER,
     .		i, i0, i1, i2, i3, eqpos, inlen, outlen,
     .          dset, nlen, maxlen

        CHARACTER*1 tab
	PARAMETER     ( tab = CHAR(9))

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
           IF ( (orig(i3:i3).LT."0" .OR. orig(i3:i3).GT."9")
     .    .AND. orig(i3:i3).NE."." ) GOTO 310
 300    CONTINUE
        GOTO 500   ! digit is final character ??
 310    i3 = i3 - 1
	dset = FIND_DSET_NUMBER(orig(i2:i3))
        IF (dset.LT.1 .OR. dset.GT.maxdsets) GOTO 10    ! error?

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
