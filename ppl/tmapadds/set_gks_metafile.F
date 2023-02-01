        SUBROUTINE SET_GKS_METAFILE

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
*       J Davison JISAO/PMEL/TMAP
*       3.15.89

*       For GKS plots (pltype = 3 or 4) captures WS type and whether to open a
*       GKS metafile, or not.  Opens GKS if not open, and activates/deactivates
*       metafile WS.

*       *JD* 4.22.91 No need for metafile wstype w new metafile translator
*       *jd* 3.10.92 Mod to handle ATC GKS methods
*       *jd* 10.7.92 Mod to use wstype default (GKS3Dwstype) in DEC
*       *jd* 2.24.93 Mod to add PPLP version numbers in metafile
*       *jd* 8.10.93 Mod to add XGKS options for WS types
*       *jd* 10.26.93 Mod to eliminate version number from xgks
*       *jd* 12.8.93 Mod to put line color rep info into metafile for xgks
*       *sh* 3.4.97 Mod to support metafile-only (no X) output through a
*               cuckoo's egg implantation of meta_wsid in place of wsid
*       *js* 7.25.97 Added 'C' callable save_metafile_name to save string
*               in variable 'meta_file'
*       *acm* 2.08 Added 'C' callable get_metafile_name to retrieve string
*               in variable 'meta_file' and return it.
* v552 *acm* 3/03 Longer strings.
* V608 *acm*10/07  true batch mode for metafiles; flag meta_batch
*         distinguises this case from other batch modes.
* V610  *acm* 3/08 Fixes for metafile batch mode
* V68   *acm* 4/11 change how flag meta_batch is computed (different Makefile)
* V6.8 *acm* 7/9/2012 For batch graphics on RHEL6, change ITS_BATCH_GRAPHICS
*                     from logical to integer, then batch = ITS_BATCH_GRAPHICS().NE.0

        IMPLICIT NONE
        include 'pltcom_dat.decl'
        include 'PLTCOM.DAT'
        include 'cmrdl_inc.decl'
        include 'CMRDL.INC'
        include 'gkscm1_inc.decl'
        include 'GKSCM1.INC'
        include 'gkscm2.cmn'
        include 'ws_types.cmn'
        INCLUDE 'parampl5_dat.decl'
        INCLUDE 'PARAMPL5.DAT'
        INCLUDE 'pen_inc.decl'  ! with num_line_colors
        INCLUDE 'PEN.INC'

        integer         beg,iend,defws,ndx,type,error
        real            red,green,blue
        character*2048  str
        character*5     kdefws

        LOGICAL xwindows

*************************************************************************

*       DETERMINE IF WINDOWING IS X OR NOT
        xwindows = .true.

*       MASSAGE COMMAND STRING OF FORM "WS=xxx,META"
        CALL UPNSQUISH(label,str,iend)

*       OPEN GKS AND WS IF NOT ALREADY OPEN
        IF ( gksopn ) GOTO 10

*       DEFINE DEFAULT WS TYPE
        CALL GETENV('XGKSwstype',kdefws)

        IF (kdefws .NE. ' ') THEN
           READ(UNIT=kdefws,FMT='(i5)') defws
        ELSE
           defws = ws_xwindow
        ENDIF

*       EXTRACT WS INFO FROM PLTYPE STRING ARGUMENT
        beg = INDEX(str,'WS')

*       USE DEFAULT IF WS TYPE NOT SPECIFIED BELOW
        IF (beg .EQ. 0) THEN
           wstype = defws
        ELSE IF ( INDEX(str(beg+3:beg+6),'VSII') .NE. 0 ) THEN
           IF (xwindows) THEN
              wstype = ws_xwindow
           ELSE 
              wstype = ws_uis
           ENDIF
        ELSE IF ( INDEX(str(beg+3:beg+9),'TEK4014') .NE. 0 ) THEN
           wstype = ws_tek4014
        ELSE IF ( INDEX(str(beg+3:beg+9),'TEK4107') .NE. 0 ) THEN
           wstype = ws_tek4107
        ENDIF
        CALL OPEN_GKS_WS

*       EXTRACT METAFILE CMD FROM PLTYPE STRING ARGUMENT
10      beg = INDEX(str,'META')
        IF ( beg .NE. 0 ) THEN
           IF ( .NOT. imgname_sent ) THEN
               CALL OPEN_METAFILE
           ENDIF
        END IF

        RETURN
        END


*
* Save the image filename in meta_file.  Set the save_on_exit
* flags to .TRUE. so the image will be saved.  Do not call
* OPEN_METAFILE at this time since this is only called on
* on start-up and there is no active window at this time.
* Instead, set imgname_sent to .FALSE.
*
        SUBROUTINE SAVE_METAFILE_NAME(mname, clen)
        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'
        INTEGER clen
        integer*1 mname(clen)

        CALL TM_CTOF_STRNG(mname, meta_file, clen)
        save_on_exit = .TRUE.
        imgname_sent = .FALSE.
        RETURN
        END


        SUBROUTINE GET_METAFILE_NAME(filename)
        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'
        CHARACTER*(*) filename
        filename = meta_file
        RETURN
        END


*
* This function is for setting save_on_exit to .FALSE. before
* starting plot_plus, thus preventing creation of an empty
* image file.
*
        SUBROUTINE CANCEL_SAVEONEXIT
        IMPLICIT NONE
        INCLUDE 'gkscm2.cmn'
        save_on_exit = .FALSE.
        RETURN
        END

