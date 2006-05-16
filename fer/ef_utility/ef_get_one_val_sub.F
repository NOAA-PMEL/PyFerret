c*
c* ef_get_one_val_sub.F
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
c*
c* Jonathan Callahan
c* Dec 17th 1997
c*
c* Revisions:
c* *sh* 10/00 - if mr_list isn't defined (as during grid creation) then
c* check to see if we can still return the value of a constant arg

c* This subroutine is called by an external function which wishes
c* to get the value of a 1x1x1x1 variable.

c* Note that grabbing a constant array during grid creation time is more
c* difficult (though not impossible). The EVAL_EXPR routine cannot safely be
c* called since the stacks would get re-initialized and the {,,,} would need
c* to be parsed


      SUBROUTINE EF_GET_ONE_VAL_SUB(id, memory, iarg, value)

      INCLUDE 'ferret.parm'
      INCLUDE 'errmsg.parm'
      INCLUDE 'xvariables.cmn'
      INCLUDE 'xcontext.cmn'
      INCLUDE 'xmem_subsc.cmn'
      INCLUDE 'EF_Util.parm'

* calling argument declarations
      REAL    value
      INTEGER id, iarg
      REAL    memory(mem_blk_size, max_mem_blks)

* internal variable declarations
      INTEGER mr, cx, uv, status
      INTEGER mr_list(EF_MAX_ARGS)
      CHARACTER buff*32

c* The following variables were cached by Ferret and must be retrieved
c* to be accessible to the external function.
      CALL EF_GET_MR_LIST(mr_list)

      IF (mr_list(1) .GT. 0 ) THEN
        mr = mr_list(iarg)
        CALL EF_GET_ONE_VAL_SUB_SUB(memory(1, mr_blk1(mr)), 
     .     lo_ss(iarg,1), hi_ss(iarg, 1),
     .     lo_ss(iarg,2), hi_ss(iarg, 2),
     .     lo_ss(iarg,3), hi_ss(iarg, 3),
     .     lo_ss(iarg,4), hi_ss(iarg, 4),
     .     value, status)   ! note that status is not valid as returned 10/00
!        IF  (status .NE. ferr_ok) GOTO 5000

      ELSE
c* memory vars havnt yet been defined -- can still try to evaluate a constant
c* Constants appear here inside a child variable
        CALL EF_GET_CX_LIST(mr_list) ! get contexts
        cx = mr_list(iarg)
        IF (cx_category(cx) .NE. cat_user_var) GOTO 5000
        uv = cx_variable(cx)
        IF (uvar_num_items(uv)   .NE. 1
     . .OR. uvar_item_type(1,uv) .NE. alg_constant ) GOTO 5000
        buff = uvar_text(uv)
        READ (buff, *, ERR=5000 ) value

      ENDIF

      RETURN

c* return a ststus word to the calling C program      
!5000 CALL ERRMSG(ferr_ef_error, status, ' ', *5100)
 5000 WRITE (buff, 3000) iarg
 3000 FORMAT ('Argument',I3,' must be a constant')
      CALL EF_BAIL_OUT(id, buff)

      RETURN 
      END
