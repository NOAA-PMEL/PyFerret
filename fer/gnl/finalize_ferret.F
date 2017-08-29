        SUBROUTINE FINALIZE_FERRET

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
*  free allocated memory held in global variables/lists
*       *acm* 3/12 cleanup ifdefs and unnecessary include files

        include 'tmap_errors.parm'  ! error codes
        include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'xio.cmn_text'
        include 'xdset_info.cmn_text'
        external xio_data
        include 'xtm_grid.cmn_text'
        external xgt_grid_data
        include 'ferret.parm'
        include 'gfdl_vms.parm'
        include 'errmsg.parm'
        include 'gfdl.parm'   ! from phil/sieg
        include 'xonedim.cmn'  ! from phil.sieg
        include 'xfr_grid.cmn'
        include 'xprog_state.cmn'
        include 'xplot_state.cmn'
        include 'xalgebra.cmn'
        include 'xvariables.cmn'
        include 'xtoday.cmn'
        include 'xpyvar_info.cmn'

C       Local variables
        INTEGER j, k, istatus, cmnd_stack_level

C       Silently process commands to clean everything up 
C       (primarily for PyFerret, but also memory leak detection)

C       Close some files that could still be open
        IF ( jrnl_lun .NE. unspecified_int4 ) THEN
            CLOSE(UNIT=jrnl_lun)
            jrnl_lun = unspecified_int4
        ENDIF

C       Reset some MODE values throughout the stack.  These actually 
C       should all be reinitialized at the end of this routine to the 
C       values in xprog_state_data.F 
        DO k = 1, mode_stack_size
            mode_state(pmode_diagnostic,k) = .FALSE.
            mode_state(pmode_verify,k) = .FALSE.
            mode_state(pmode_journal,k) = .FALSE.
        ENDDO

        cmnd_buff = 'CANCEL REDIRECT'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *100)
        CALL XEQ_CANCEL
  100   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        cmnd_buff = 'SET GRID ABSTRACT'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *200)
        CALL XEQ_SET
  200   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        IF ( pplus_started ) THEN
            cmnd_buff = 'CANCEL WINDOW /ALL'
            CALL GET_FER_COMMAND(cmnd_buff, istatus, *300)
            CALL XEQ_CANCEL
  300       CALL CLEANUP_LAST_CMND(cmnd_stack_level)
        ENDIF

        cmnd_buff = 'CANCEL VARIABLE /ALL'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *400)
        CALL XEQ_CANCEL
  400   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        cmnd_buff = 'CANCEL ALIAS /ALL'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *500)
        CALL XEQ_CANCEL
  500   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        cmnd_buff = 'CANCEL LIST /ALL'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *600)
        CALL XEQ_CANCEL
  600   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        cmnd_buff = 'CANCEL REGION /ALL'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *700)
        CALL XEQ_CANCEL
  700   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

        cmnd_buff = 'CANCEL SYMBOL /ALL'
        CALL GET_FER_COMMAND(cmnd_buff, istatus, *800)
        CALL XEQ_CANCEL
  800   CALL CLEANUP_LAST_CMND(cmnd_stack_level)

C       "CANCEL DATA /ALL"
C       Code pulled from xeq_cancel
        DO k = 1, maxdsets
           IF ( ds_name(k) .NE. char_init2048 ) THEN
              CALL CANCEL_ONE_DATA_SET(k, istatus)
           ENDIF
        ENDDO

C       "CANCEL GRID /ALL"
C       Code pulled from xeq_cancel in order to do complete clean up silently
        DO k = 1, grid_ceiling
           IF (grid_name(k) .NE. char_init16) THEN
              grid_use_cnt(k) = 0
              IF ( k .LE. max_grids ) THEN
                 grid_name(k) = char_init16
              ELSE
                 CALL TM_DEALLO_DYN_GRID(k)
              ENDIF
           ENDIF
        ENDDO

C       "CANCEL AXIS /ALL"
C       Code pulled from DEALLO_ALL_AXES in order to do complete clean up silently
        DO k = 1, line_ceiling
           IF (line_name(k) .NE. char_init16) THEN
              line_use_cnt(k) = 0
              line_keep_flag(k) = .FALSE.
              IF ( k .LE. max_lines ) THEN
                 IF ( .NOT. line_regular(k) ) 
     .              CALL PACK_LINE_STORAGE(k)
                 line_name(k) = char_init16
              ELSE
                 CALL TM_DEALLO_DYN_LINE(k)
              ENDIF
           ENDIF
        ENDDO

C       "CANCEL MEMORY /ALL"
C       pulled from xeq_cancel in order to do complete clean up silently
        DO k = 1, max_mr_avail
            IF ( mr_protected(k) .NE. mr_deleted ) THEN
                IF ( mr_protected(k) .EQ. mr_perm_protected ) THEN
                    permanent_mem = permanent_mem - mr_size(k)
                ENDIF
                CALL DELETE_VARIABLE(k)
            ENDIF
        ENDDO
        peak_mem = 0
        peak_essential_mem = 0

        CALL us2i_string_list_free()

        CALL efcn_list_clear()

        CALL string_array_clear(alg_pvar_head)

        CALL string_array_clear(countervar_name_head)

        CALL string_array_clear(uvar_name_code_head)
 
        CALL string_array_clear(ds_var_code_head)

        CALL string_array_clear(pyvar_code_head)

        CALL deleted_list_clear(uvar_num_items_head)

        CALL ncf_datasets_list_clear()

        RETURN

        END

