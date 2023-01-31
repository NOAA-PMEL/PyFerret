	SUBROUTINE compute_histo_bins (z, wrk, badz, nsize, status)
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

* For histogram-based levels, compute the histogram bins. Sort the data and 
* compute the color levels so that each color band contains approximately 
* equal numbers of data points. This form of color management best brings out 
* the fine structure in the data. 
* V630  *acm* 9/09 Introduction of syntax for variance-based and histogram levels
* V68  *acm* 1/12  changes for double-precision ferret, single-precision pplus
* V691 *acm* 5/14  Define symbols containing histogram levels

        IMPLICIT NONE
        include 'parampl5_dat.decl'
        include 'PARAMPL5.DAT'
	include 'hd_inc.decl'
	include 'HD.INC'
	include 'cont_inc.decl'
	include 'CONT.INC'

* nlev	number of levels to use, from cont.inc
* zmin	in hd.inc, min and max of the data
* zmax

* calling argument declarations:
	INTEGER nsize, status
	REAL*8 z(*), wrk(*), badz

* internal variable declarations:

      INTEGER i, ngood, ibin, n_per_bin

      CHARACTER ppl_buff*2048, TM_FMT_SNGL*48, buff*10
      INTEGER TM_LENSTR1, ier, ilen, slen, ist
      LOGICAL TM_FPEQ_SNGL

      ngood = 0
      DO i = 1, nsize
         wrk(i) = z(i)
         IF (wrk(i) .NE. badz) ngood = ngood + 1
      ENDDO

      CALL HEAP2_V (wrk, badz, nsize)

* nlev refers to the edges of the color bins; make the
* user's input refer to the number of bins.

      nlev = nlev + 1  
      nlev2 = nlev
      IF (ngood .LT. nlev) nlev2 = ngood

      n_per_bin = ngood/ (nlev2-1)

      ibin = 1
      DO i = 1, ngood, n_per_bin
         zlev(ibin) = wrk(i)
         ibin = ibin + 1
      ENDDO
      zlev(1) = wrk(1)
      zlev(nlev2) = wrk(ngood)

*      ... could smooth these some.

      IAUTOC = 0  ! levels are already set so dont do it in set_levels.

      USE_HIST = .TRUE.

* Define the LEV_HISTO_LEVELS symbol here. It may be wanted in a PLOT/SET command
* before setlab_syms would be executed.

       IF (USE_HIST) THEN  
          ppl_buff = 'LEV_HISTO_LEVELS '
	  ilen = TM_LENSTR1(ppl_buff)+1
          DO 200 i =1, nlev2
	     buff = TM_FMT_SNGL(zlev(i), -2, 10, slen)
             ppl_buff = ppl_buff(:ilen)//buff(:slen)
	     ilen = TM_LENSTR1(ppl_buff)
	     IF (i.LT.nlev2) ppl_buff = ppl_buff(:ilen)//','
	     ilen = TM_LENSTR1(ppl_buff)
  200     CONTINUE
         ier = 0
         ist = 0
         CALL SETSYM (ppl_buff, ilen, ier, ist)
      ENDIF

      RETURN
      END
