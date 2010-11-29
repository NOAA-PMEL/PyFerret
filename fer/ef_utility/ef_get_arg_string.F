c*
c* ef_get_arg_string.F
c*   
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
c* Jonathan Callahan
c* Mar 17th 1999
* V530  8/00 *sh* - shifted to new way of storing strings
*      10/00      - allow this routine to be called during CUSTOM AXIS time
*                      (before evaluation of variables)
c*
c*v540 1/02 *kob* - fix bug which used character array risc_buff to pass
c*                  a string into tm_ctof_string when tm_ctof_string was
c*                  expecting an integer array.  no longer need xrisc.cmn

c*
c* This returns the (single) string which was passed as an argument
c* to the function.  
c* It may be called during grid creation time (custom axis creation) or
c* during evaluation time.
c* Other routines are available to handle arrays of strings.
c* Note that grabbing an array of string during grid creation time is more
c* difficult (though not impossible). The EVAL_EXPR routine cannot safely be
c* called since the stacks would get re-initialized and the {,,,} would need
c* to be parsed
c*

      SUBROUTINE EF_GET_ARG_STRING(id, iarg, text)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'EF_Util.parm'

      INTEGER id, iarg, mr
      CHARACTER*(*) text

      INTEGER  GET_OFFSET_C_STRING_LEN, slen, flen, cx, uv,
     .         item, start, end
      INTEGER mr_list(EF_MAX_ARGS)
      CHARACTER buff*32
 
c The memory resident list was cached by Ferret and must be retrieved
c to be accessible to the external function.
      CALL EF_GET_MR_LIST(mr_list)
      mr = mr_list(iarg)
      
      IF ( mr .EQ. 0 ) THEN
* we must have been called during custom axis creation time. We'll need
* to grab the string straight from the function definition
* (e.g. EF("my_string") and also allow A="string"; EF(a)
        CALL EF_GET_CX_LIST(mr_list) ! get contexts
        cx = mr_list(iarg)
        IF (cx_category(cx) .EQ. cat_user_var) THEN
* ... an intermediate variable was defined as a constant string
          uv = cx_variable(cx)
          IF (uvar_num_items(uv)   .NE. 1
     .   .OR. uvar_item_type(1,uv) .NE. alg_string ) GOTO 5000
          item = 1
        ELSEIF (cx_category(cx) .EQ. cat_string) THEN
* ... determine the string position directly within EF call
          uv = cx_variable(cx)/1000
          item = cx_variable(cx) - 1000*uv
        ELSE
          GOTO 5000
        ENDIF
* remove the surrounding quotation marks
        start = uvar_item_start( item, uv ) + 1 ! +1 to bypass quotations
        end   = uvar_item_end  ( item, uv ) - 1
        text = uvar_text(uv)(start:end)

      ELSE

c Here's how you check the variable type.
* (this should never happen as checks against it are made previously) 
        IF (mr_type(mr) .NE. ptype_string) WRITE (6,19) iarg
 19     FORMAT(' ERR: non-string passed as string arg', I2)

* get the string
        slen = GET_OFFSET_C_STRING_LEN(mr_c_pointer(mr), 0)

* get the a null-terminated C string
        CALL GET_OFFSET_C_STRING(mr_c_pointer(mr), 0,
     .                  int_buff, 2048)
        
* the longest length we can use ...
        flen = MIN(slen, size_ibuff)
        flen = MIN(flen, LEN(text))
        
* copy it into the FORTRAN string
        CALL TM_CTOF_STRNG (int_buff, text, flen)

      ENDIF

      RETURN

c* return a ststus word to the calling C program      
 5000 WRITE (buff, 3000) iarg
 3000 FORMAT ('Argument',I3,' is not a string')
      CALL EF_BAIL_OUT(id, buff)

      END

