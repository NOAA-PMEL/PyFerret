*
* labwid.F
*
* Ansley Manke
* March 1, 2005
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration''s (NOAA) Pacific Marine Environmental Lab(PMEL),
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
* Get the width of a label in plot inches.
*
* This function calls the PPLUS routine SYMWID.  It is not an external 
* function, will be linked internally; but it is convenient to implement
* it as an external function.

* V69+  5/14 *acm* Fix for ascii font in multi-line labels, ticket 2176

      SUBROUTINE labwid_init(id)

* Define arguments and result

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id, arg

      CALL ef_set_desc(id, 'Return string width in PLOT inches' )
      CALL ef_set_num_args(id, 2)
      CALL ef_set_axis_inheritance_6d(id,
     .                                ABSTRACT, NORMAL,
     .                                NORMAL,   NORMAL,
     .                                NORMAL,   NORMAL)
      CALL ef_set_piecemeal_ok_6d(id, NO, NO, NO, NO, NO, NO)

      arg = 1
      CALL ef_set_arg_name(id, arg, 'STR')
      CALL ef_set_arg_desc(id, arg, 'strng, including font')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)
      CALL ef_set_arg_type(id, arg, string_arg)

      arg = 2
      CALL ef_set_arg_name(id, arg, 'HT')
      CALL ef_set_arg_desc(id, arg, 'Size of character')
      CALL ef_set_axis_influence_6d(id, arg,
     .                              NO, NO, NO, NO, NO, NO)

      RETURN 
      END


* Define abstract output axis: 1 value

      SUBROUTINE labwid_result_limits(id)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'

      INTEGER id
      INTEGER ivalue

      ivalue = 1
      CALL ef_set_axis_limits(id, X_AXIS, ivalue, ivalue)

      RETURN 
      END


*
* Compute the result
*
      SUBROUTINE labwid_compute(id, arg_1, arg_2, result)

      IMPLICIT NONE
      INCLUDE 'EF_Util.cmn'
      INCLUDE 'EF_mem_subsc.cmn'


      INTEGER id

      REAL arg_1(mem1lox:mem1hix, mem1loy:mem1hiy, mem1loz:mem1hiz, 
     .           mem1lot:mem1hit, mem1loe:mem1hie, mem1lof:mem1hif)
      REAL arg_2(mem2lox:mem2hix, mem2loy:mem2hiy, mem2loz:mem2hiz, 
     .           mem2lot:mem2hit, mem2loe:mem2hie, mem2lof:mem2hif)

      REAL result(memreslox:memreshix, memresloy:memreshiy, 
     .            memresloz:memreshiz, memreslot:memreshit,
     .            memresloe:memreshie, memreslof:memreshif)

* After initialization, the 'res_' arrays contain indexing information 
* for the result axes.  The 'arg_' arrays will contain the indexing 
* information for each variable''s axes. 

      INTEGER res_lo_ss(6),
     .        res_hi_ss(6),
     .        res_incr (6)
      INTEGER arg_lo_ss(6,EF_MAX_ARGS),
     .        arg_hi_ss(6,EF_MAX_ARGS),
     .        arg_incr (6,EF_MAX_ARGS)

      INTEGER arg, i, j, k, l, m, n, status
      REAL    LABEL_WIDTH, height
      CHARACTER strng*1024, errtxt*250

      CALL ef_get_res_subscripts_6d(id, res_lo_ss, res_hi_ss, res_incr)

      arg = 1
      CALL ef_get_arg_string(id, arg, strng)
      arg = 2
      CALL ef_get_one_val(id, arg, height)

* Compute the result.
      i = res_lo_ss(X_AXIS)
      j = res_lo_ss(Y_AXIS)
      k = res_lo_ss(Z_AXIS)
      l = res_lo_ss(T_AXIS)
      m = res_lo_ss(E_AXIS)
      n = res_lo_ss(F_AXIS)

      result(i,j,k,l,m,n) = LABEL_WIDTH(height, strng, errtxt, status)

      IF (status .NE. 1) CALL EF_BAIL_OUT(id, errtxt)

      RETURN 
      END


      REAL FUNCTION LABEL_WIDTH(height,strng, errtxt, status)

      IMPLICIT NONE
      INTEGER	status
      REAL	height
      CHARACTER*(*) strng, errtxt


      INCLUDE 'xrisc.cmn'
      include 'pyfonts.cmn'
      INCLUDE 'fgrdel.cmn'


      INTEGER	n, lnbeg(500), lnend(500), numlines,
     .		npen, nfont, npnew, nfnew, nchr, mywindowid
      REAL*4	SYMWID, ht, ss, xmn
      REAL*8	mywindow
      CHARACTER*3 fontprefix, penprefix, newfont, newpen

      CALL TM_BREAK_LINES (strng, lnbeg, lnend, numlines)
      nfont = 0
      npen = 0
      fontprefix = ""
      penprefix = ""

      ss = 0.
      ht = height

      status = 1

c PyFerret fonts need an active window to return font size info
      IF (pyfont) THEN
         
        mywindowid = ACTIVEWINDOW
C       Some sanity checks on the window ID and window
        IF ( (mywindowid .LT. 1) .OR. 
     .       (mywindowid .GT. MAXWINDOWOBJS) ) THEN
            errtxt = 'LABWID: PyFerret font queries need an active plot window'
	    status = 0
	    RETURN
        ENDIF
        mywindow = WINDOWOBJS(mywindowid)
        IF ( mywindow .EQ. NULLOBJ ) THEN
            errtxt = 'LABWID: PyFerret font queries need an active plot window'
	    status = 0
	    RETURN
        ENDIF
      ENDIF

      DO 40 n = 1, numlines
         CALL getlabfonts (strng, lnbeg, lnend, n, fontprefix, 
     .           npnew, nfnew, newfont, newpen, nchr)

         risc_buff = strng(lnbeg(n):lnend(n))


         xmn = SYMWID(ht,nchr+npen+nfont,
     .           penprefix(1:npen)//fontprefix(1:nfont)//
     .           risc_buff)

         ss = MAX(xmn, ss)

C  Set up to use any font settings from this line on subsequent lines.

         IF (npnew.GT.0) THEN
            penprefix(1:npnew) = newpen(1:npnew)
            npen = npnew
         ENDIF
         IF (nfnew.GT.0) THEN
            fontprefix(1:npnew) = newfont(1:npnew)
            nfont = npnew
         ENDIF
  40  CONTINUE

      LABEL_WIDTH = ss

      RETURN
      END

