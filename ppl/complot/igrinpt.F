      FUNCTION IGRNPT(window_id, show_menu, X,Y)
C** 
C**    @(#)igrinpt.F	1.2    3/27/88
*
*  Note for PyFerret: igrnpt is not called from Fortran but from Python
*
*    modified 6/99 for Ferret V500 *sh* -- added new arguments and
*       return COMMON XGRINPT with information about what was digitized
*       New arguments:
*               show_menu
*    Note: this routine should really be split into two -- one to put up
*       the choice menu, another to get the position input. Taking the lazy
*       approach now ...
*
* v5.11 *kob* 5/00 - added ifdef to check for f90 - if so have to
*                    initialize has_new_mouse w/in block data since it
*                    exists in common
C**
C**
C***********************************************************************
C**
C**                 PLOT+ Scientific Graphics System
C**
C***********************************************************************
C**
C**
C
C     IGRNPT(X,Y)
C
C     THIS FUNCTION ALLOWS THE USER TO INPUT GRAPHICS INFORMATION
C     FROM A TEKTRONIX GRAPHICS TERMINAL TO HIS PROGRAM.  ON ALL
C     TEKTERMINALS OTHER THAN THE PLOTTERS, WHEN IGRNPT IS
C     CALLED, A FULL-SCREEN CROSS-HAIR CURSOR APPEARS.  THE CROSS
C     HAIR MAY BE POSITIONED ON THE SCREEN BY USE OF THE THUMB-
C     WHEELS OR JOYSTICK AS APPROPRIATE.  THE USER CAN CAUSE THE
C     COORDINATES OF THE CROSS HAIR TO BE SENT TO IGRNPT BY EN-
C     TERING A CHARACTER FROM THE KEYBOARD.  THE ASCII CODE FOR
C     THIS CHARACTER IS RETURNED AS THE VALUE OF THE FUNCTION, AND
C     THE COORDINATES IN USER'S UNITS ARE RETURNED IN THE PARAM-
C     ETERS.  ON THE TEKTRONIX PLOTTER, WHEN IGRNPT IS CALLED,
C     THE PROMPT LIGHT (4662) OR POINT LIGHT (4663) IS TURNED ON.
C     IF NECESSARY, THE PEN MAY BE POSITIONED BY USE OF THE JOY-
C     STICK.  THE USER CAN CAUSE THE COORDINATES OF THE PEN TO BE
C     SENT TO IGRNPT BY MOMENTARILY DEPRESSING THE CALL BUTTON
C     (4662) OR A POINT FUNCTION (4663).  THE PEN POSITION (0 =
C     UP, 1 = DOWN) IS RETURNED AS THE VALUE OF THE FUNCTION, AND
C     THE COORDINATES IN USER'S UNITS ARE RETURNED IN THE PARAM-
C     ETERS.
C
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
*	Mods J Davison 3.21.94 for xgks
*	1) Change PET to 3
*	2) Change echo area for xgks coords in GINCH and not call GINLC
* 	*jd* linux port 12.96 change hp -> NEED_IAND for CPP choices


C V6.3 *acm* 9/09 Allow mouse clicks in windows other than window 1.
C            Note we get wrong rewults if the plot is in a viewport.
*  12/2022 *acm* Code cleanup: remove code having to do with Tektronix
*		 and other terminals. 
*		 Note that FGD calls are apparently just a 
*		 sketch for getting user input; the routines themselves
*		 have comments "input functions are not supported"


        LOGICAL show_menu
	INTEGER window_id

C	GKS DATA FOR CHOICE AND LOCATOR INPUT

	INTEGER*4 SIZES(5),PET
	INTEGER*4 ERRIND,MODE,ESW,LDR,ISTAT,ICHOICE
	INTEGER*4 XFORM,WSTYPE,DCUNIT,NULLI
	INTEGER   CHCE(5)
	CHARACTER*10 STRING(5)
	CHARACTER*80 RECORD(10),CONID
	REAL EAREA(4),LOCX,LOCY,WIND(4),VIEW(4),NULLR

	CHARACTER*1 IMODE
	DIMENSION IN(8)

	include 'PLTCOM.DAT'
        include 'xgrinpt.cmn'
        integer iwkid

	DATA CHCE/70,76,82,67,32/

        DATA has_new_mouse/ .FALSE. /   ! at program initialization


CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

C     SET ALPHAS MODE.

      IMODE = PMODE
      PMODE = ALPHA

C     IF THE LAST POINT WAS NOT PLOTTED BECAUSE THE PEN WAS UP,
C     PLOT THE POINT NOW.

      IF(SAVEF)CALL XYZPLT

C     RESTORE THE CURRENT PLOTTING MODE.

      PMODE = IMODE

C     CHECK THE PLOT TYPE.

      IF(PTYPE.GE.3)THEN

C GKS DEVICE

	CALL PPLGFLUSH
	iwkid = window_id  
	CALL FGD_GQCHS(iwkid,1,10,ERRIND,MODE,ESW,ISTAT,ICHOICE,
     * PET,EAREA,LDR,RECORD)
	STRING(5) = 'Left'
	STRING(4) = 'Center'
	STRING(3) = 'Right'
	STRING(2) = 'Line'
	STRING(1) = 'Arrow'
	SIZES(5) = 4
	SIZES(4) = 6
	SIZES(3) = 5
	SIZES(2) = 4
	SIZES(1) = 5

	PET = 3
	EAREA(1)=10.0
	EAREA(2)=250.0
	EAREA(3)=10.0
	EAREA(4)=250.0

	LDR=5
	IRECSZ=10

	CALL FGD_GPREC(0,NULLI,0,NULLR,5,SIZES,STRING,IRECSZ,
     * ISTAT,LDR,RECORD)
	CALL FGD_GINCH(iwkid,1,1,5,PET,EAREA(1),EAREA(2),EAREA(3),
     * EAREA(4),LDR,RECORD)

	IF (show_menu) THEN     ! **************
	   CALL FGD_GRQCH(iwkid,1,ISTAT,ICHOICE)       ! put up menu and get answer
	ELSE
	   ichoice = 5         ! always left justified
	ENDIF                   ! **************

	IGRNPT=CHCE(ICHOICE)
	CALL FGD_GSVPIP(1,0,0)
	CALL FGD_GQLCS(iwkid,1,1,7,ERRIND,MODE,ESW,XFORM,
     *                LOCX,LOCY,PET,EAREA,LDR,RECORD)
	PET = 3
*           Note that FGD_GQDSP uses the window id 
*           instead of window type as the first argument
	CALL FGD_GQDSP(iwkid,ERRIND,DCUNIT,DCX,DCY,IRSX,IRSY)
	EAREA(2)=DCX
	EAREA(4)=DCY

	CALL FGD_GSLCM(iwkid,1,0,1)
	CALL FGD_GRQLC(iwkid,1,ISTAT,XFORM,LOCX,LOCY)  ! get mouse click (X,Y)
	CALL FGD_GQNT(XFORM,ERRIND,WIND,VIEW)
	AOLD= LOCX
	BOLD= LOCY
	ENDIF

	IF(.NOT.(IABS(PTYPE).EQ.2.OR.PTYPE.EQ.1))RETURN

	RETURN
	END
