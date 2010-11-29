      SUBROUTINE setax_syms(dir, axlo, axhi)
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
*  v5.41  *acm* set XAXIS_MIN, XAXIS_MAX, YAXIS_MIN, YAXIS_MAX if 
*               they have not already been set.  Called by setax.
** v552 *acm* 3/03 Longer strings: changes for multi-line labels.
*  v552 *acm* 4/03 buff was too short a string
*  v608 *acm* 10/07 Formatting very small numbers resulted in 0.
*               (bug 1542). Use same formatting as in axis_end_syms.F

* Calling argument declarations

      REAL axlo, axhi
      CHARACTER*1 dir

* Internal arguments

      CHARACTER axval*255, ppl_buff*2048, buff*30
      INTEGER TM_LENSTR1, nc, ier, ilen 

      buff = dir//'AXIS_MIN'
      CALL GETSYM(buff,axval,nc,ier)
      IF (ier .EQ. 1) THEN
         WRITE ( buff, '(1PG15.7)' ) axlo
         ppl_buff = dir // 'AXIS_MIN '// buff
         ier = 0
         ist = 0
         ilen = TM_LENSTR1(ppl_buff)
         CALL SETSYM (ppl_buff, ilen, ier, ist)
      ENDIF

      buff = dir//'AXIS_MAX'
      CALL GETSYM(buff,axval,nc,ier)
      IF (ier .EQ. 1) THEN
         WRITE ( buff, '(1PG15.7)' ) axhi
         ppl_buff = dir // 'AXIS_MAX '// buff
         ier = 0
         ist = 0
         ilen = TM_LENSTR1(ppl_buff)
         CALL SETSYM (ppl_buff, ilen, ier, ist)
      ENDIF

      RETURN
      END
