      SUBROUTINE setlab_syms (zlev, nlev)
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
*  v5.5  *acm* 11/2002 New symbols to save LABEL levels

*    LEV_MIN  - lowest level used on last CONTOUR, SHADE, or FILL call.
*    LEV_MAX -  highest level used last
*    LEV_DEL - either the value or "irregular"
*    LEV_NUM - Number of levels used
*    LEV_ALL - ?? list of all the levels used

*    (LEV_TEXT - full text string of last levels used, set in routine LEV.F)

* Calling argument declarations

      REAL zlev(*)
      INTEGER nlev

* Internal arguments

      REAL zmax, zmin, dz

      CHARACTER ppl_buff*255, TM_FMT*10, buff*10
      INTEGER TM_LENSTR1, i, ier, ilen, slen 
      LOGICAL TM_FPEQ, reglevs

* Lowest and highest Z level

      zmin = zlev(1)
      buff = TM_FMT(zmin, -2, 10, slen)

      ppl_buff = "LEV_MIN "//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)

      zmax = zlev(nlev)
      buff = TM_FMT(zmax, -2, 10, slen)

      ppl_buff = "LEV_MAX "//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)

* Number of levels used 

      zmax = nlev
      buff = TM_FMT(zmax, -2, 10, slen)

      ppl_buff = "LEV_NUM "//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)
      
* Check if levels are regularly spaced 

      dz = zlev(2) - zlev(1)
      
      IF (nlev .EQ. 1) THEN
         buff = 'none'
      ELSE IF (nlev .EQ. 2) THEN
         buff = TM_FMT(dz, -2, 10, slen)
      ELSE
         reglevs = .TRUE.
         DO 100 i=3,nlev
            IF(.NOT. TM_FPEQ(dz,(zlev(i)-zlev(i-1))) ) reglevs=.FALSE.
  100    CONTINUE
         IF (reglevs) THEN
            buff = TM_FMT(dz, -2, 10, slen)
         ELSE
            buff = 'irregular'
         ENDIF
      ENDIF

      ppl_buff = "LEV_DEL "//buff
      ilen = TM_LENSTR1(ppl_buff)
      ier = 0
      ist = 0
      CALL SETSYM (ppl_buff, ilen, ier, ist)

      RETURN
      END
