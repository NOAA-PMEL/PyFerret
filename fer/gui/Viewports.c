/*
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
*/




/*******************************************************************************
	Viewports.c

       Associated Header file: Viewports.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/ToggleB.h>
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/Frame.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ferret_structures.h"
#include "vpll.xpm"
#include "vp_cycle_corners_in.xpm"
#include "vpll_sel.xpm"
#include "vplower.xpm"
#include "vplower_sel.xpm"
#include "vp_cycle_corners_out.xpm"
#include "vplr.xpm"
#include "vp_cycle_corners_out_ins.xpm"
#include "vpfull_sel.xpm"
#include "vpfull.xpm"
#include "vplr_sel.xpm"
#include "vp_cycle_lr_in.xpm"
#include "vpright.xpm"
#include "vpright_sel.xpm"
#include "vp_cycle_lr_out.xpm"
#include "vpul.xpm"
#include "vp_cycle_lr_out_ins.xpm"
#include "vpul_sel.xpm"
#include "vpupper.xpm"
#include "vpupper_sel.xpm"
#include "vpleft.xpm"
#include "vpur.xpm"
#include "vpleft_sel.xpm"
#include "vpur_sel.xpm"

/* globals */
swidget gSavedViewPorts = NULL;
swidget Viewports;
extern Boolean gHiRez;

static void ClearAllViewPorts(void);
static void ClearCycleButtons(void);
extern void ferret_command(char *cmdText, int cmdMode);
swidget create_Create_Viewports(swidget UxParent);
static void JC_ClearButton_CB(void);
extern void MaintainMainWdBtns(void);
extern Pixmap GetPixmapFromData(char **inData);
extern Pixmap GetBitmapFromData(char *inData, unsigned width, unsigned height);
static void InitPixmaps();

extern int gViewportActive, gViewportIsCycling, gCurrViewportType, 
	gNumViewportCycles[3], gCurrViewportCycle, gSomethingIsPlotted;
static int gCycleState=0;


static	Widget	form8;
static	Widget	label69;
static	Widget	form23;
static	Widget	rowColumn12;
static	Widget	toggleButton37;
static	Widget	toggleButton38;
static	Widget	toggleButton39;
static	Widget	toggleButton40;
static	Widget	toggleButton41;
static	Widget	toggleButton42;
static	Widget	toggleButton43;
static	Widget	toggleButton44;
static	Widget	toggleButton45;
static	Widget	toggleButton62;
static	Widget	toggleButton63;
static	Widget	toggleButton64;
static	Widget	rowColumn26;
static	Widget	pushButton30;
static	Widget	pushButton31;
static	Widget	pushButton32;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "Viewports.h"
#undef CONTEXT_MACRO_ACCESS

Widget	Viewports;
Widget	frame14;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(toggleButton62),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_corners_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_corners_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_corners_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton63),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton64),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vp_cycle_lr_out_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(vp_cycle_lr_out_ins_xpm),
		XmNselectPixmap, GetPixmapFromData(vp_cycle_lr_in_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton37),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpfull_xpm),
		XmNselectPixmap, GetPixmapFromData(vpfull_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton38),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpll_xpm),
		XmNselectPixmap, GetPixmapFromData(vpll_sel_xpm),
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton39),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplr_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplr_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);  

	XtVaSetValues(UxGetWidget(toggleButton40),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpur_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpur_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL); 

	XtVaSetValues(UxGetWidget(toggleButton41),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpul_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpul_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton42),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpleft_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpleft_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton43),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpright_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpright_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton44),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vpupper_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vpupper_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);

	XtVaSetValues(UxGetWidget(toggleButton45),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(vplower_xpm),
		/*XmNlabelInsensitivePixmap, GetPixmapFromData() */
		XmNselectPixmap, GetPixmapFromData(vplower_sel_xpm),
		/*XmNselectInsensitivePixmap, GetPixmapFromData(), */
		NULL);
}

static void JC_ClearButton_CB()
{
	ferret_command("SET WINDOW/CLEAR", IGNORE_COMMAND_WIDGET);
	/*gSomethingIsPlotted = 0;
	MaintainMainWdBtns(); */
}

static void InitInterface()
{
	/* initialize the viewports */
	ClearAllViewPorts();

	/* cycle buttons insensitive */
	ClearCycleButtons();

	if (gViewportActive) {
		switch (gCurrViewportType) {
			case 0:
				switch (gCurrViewportCycle) {
					case 0:
						XmToggleButtonSetState(UxGetWidget(toggleButton38), True, False);
						break;
					case 1:
						XmToggleButtonSetState(UxGetWidget(toggleButton39), True, False);
						break;
					case 2:
						XmToggleButtonSetState(UxGetWidget(toggleButton40), True, False);
						break;
					case 3:
						XmToggleButtonSetState(UxGetWidget(toggleButton41), True, False);
						break;
				}
				if (gViewportIsCycling)
					XmToggleButtonSetState(UxGetWidget(toggleButton62), True, False);
				break;
			case 1:
				switch (gCurrViewportCycle) {
					case 0:
						XmToggleButtonSetState(UxGetWidget(toggleButton42), True, False);
						break;
					case 1:
						XmToggleButtonSetState(UxGetWidget(toggleButton43), True, False);
						break;
				}
				if (gViewportIsCycling)
					XmToggleButtonSetState(UxGetWidget(toggleButton63), True, False);
				break;
			case 2:
				switch (gCurrViewportCycle) {
					case 0:
						XmToggleButtonSetState(UxGetWidget(toggleButton44), True, False);
						break;
					case 1:
						XmToggleButtonSetState(UxGetWidget(toggleButton45), True, False);
						break;
				}
				if (gViewportIsCycling)
					XmToggleButtonSetState(UxGetWidget(toggleButton64), True, False);
				break;
		}
	}
	else {
		/* full window selected */
		XmToggleButtonSetState(UxGetWidget(toggleButton37), True, False);
	}
}

static void ClearCycleButtons()
{
	XmToggleButtonSetState(UxGetWidget(toggleButton62), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton63), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton64), False, False);
}

static void ClearAllViewPorts()
{
	XmToggleButtonSetState(UxGetWidget(toggleButton37), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton38), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton39), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton40), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton41), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton42), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton43), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton44), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton45), False, False);
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	createCB_Viewports(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	
	}
}

static	void	popupCB_Viewports(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	InitInterface();
}

static	void	destroyCB_Viewports(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedViewPorts = NULL;
}

static	void	valueChangedCB_toggleButton37(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		ClearCycleButtons();
		gCurrViewportType = -1;
		if (gViewportActive) {
			ferret_command("CANCEL VIEWPORT", IGNORE_COMMAND_WIDGET);
			gViewportActive = False;
		}
	}
	}
}

static	void	armCB_toggleButton37(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton38(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 1) {
			ClearCycleButtons();
			gCycleState = 1;
		}
		gViewportActive = True;
		gCurrViewportType = 0;
		gCurrViewportCycle = 0;
	}
	
	}
}

static	void	armCB_toggleButton38(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton39(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 1) {
			ClearCycleButtons();
			gCycleState = 1;
		}
		gViewportActive = True;
		gCurrViewportType = 0;
		gCurrViewportCycle = 1;
	}
	}
}

static	void	armCB_toggleButton39(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton40(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 1) {
			ClearCycleButtons();
			gCycleState = 1;
		}
		gViewportActive = True;
		gCurrViewportType = 0;
		gCurrViewportCycle = 2;
	}
	}
}

static	void	armCB_toggleButton40(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton41(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 1) {
			ClearCycleButtons();
			gCycleState = 1;
		}
		gViewportActive = True;
		gCurrViewportType = 0;
		gCurrViewportCycle = 3;
	}
	}
}

static	void	armCB_toggleButton41(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton42(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 2) {
			ClearCycleButtons();
			gCycleState = 2;
		}
		gViewportActive = True;
		gCurrViewportType = 1;
		gCurrViewportCycle = 0;
	}
	}
}

static	void	armCB_toggleButton42(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton43(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 2) {
			ClearCycleButtons();
			gCycleState = 2;
		}
		gViewportActive = True;
		gCurrViewportType = 1;
		gCurrViewportCycle = 1;
	}
	}
}

static	void	armCB_toggleButton43(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton44(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 3) {
			ClearCycleButtons();
			gCycleState = 3;
		}
		gViewportActive = True;
		gCurrViewportType = 2;
		gCurrViewportCycle = 0;
	}
	}
}

static	void	armCB_toggleButton44(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton45(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		if (gCycleState != 3) {
			ClearCycleButtons();
			gCycleState = 3;
		}
		gViewportActive = True;
		gCurrViewportType = 2;
		gCurrViewportCycle = 1;
	}
	}
}

static	void	armCB_toggleButton45(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* arm */
	}
}

static	void	valueChangedCB_toggleButton62(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		XmToggleButtonSetState(UxGetWidget(toggleButton63), False, False);
		XmToggleButtonSetState(UxGetWidget(toggleButton64), False, False);
		/* is this the first click in this group */
		if (gCurrViewportType != 0) {
			gViewportActive = True;
			gCurrViewportCycle = 0;
			gCurrViewportType = 0;
			ClearAllViewPorts();
			XmToggleButtonSetState(UxGetWidget(toggleButton38), True, False);
		}
		
		gViewportIsCycling = True;
		if (gViewportActive) 
			ferret_command("CANCEL VIEWPORT", IGNORE_COMMAND_WIDGET);
	}
	else
		gViewportIsCycling = False;
	
	}
}

static	void	valueChangedCB_toggleButton63(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		XmToggleButtonSetState(UxGetWidget(toggleButton62), False, False);
		XmToggleButtonSetState(UxGetWidget(toggleButton64), False, False);
		gViewportIsCycling = True;
		/* is this the first click in this group */
		if (gCurrViewportType != 1) {
			gViewportActive = True;
			gCurrViewportType = 1;
			gCurrViewportCycle = 0;
			ClearAllViewPorts();
			XmToggleButtonSetState(UxGetWidget(toggleButton42), True, False);
		}
		if (gViewportActive) 
			ferret_command("CANCEL VIEWPORT", IGNORE_COMMAND_WIDGET);
	}
	else
		gViewportIsCycling = False;
	
	}
}

static	void	valueChangedCB_toggleButton64(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	/* value changed */
	XmToggleButtonCallbackStruct *cbInfo = (XmToggleButtonCallbackStruct *)UxCallbackArg;
	
	if (cbInfo->set) {
		XmToggleButtonSetState(UxGetWidget(toggleButton62), False, False);
		XmToggleButtonSetState(UxGetWidget(toggleButton63), False, False);
		/* is this the first click in this group */
		if (gCurrViewportType != 2) {
			gViewportActive = True;
			gCurrViewportType = 2;
			ClearAllViewPorts();
			gCurrViewportCycle = 0;
			XmToggleButtonSetState(UxGetWidget(toggleButton44), True, False);
		}
		gViewportIsCycling = True;
		if (gViewportActive) 
			ferret_command("CANCEL VIEWPORT", IGNORE_COMMAND_WIDGET);
	}
	else
		gViewportIsCycling = False;
	
	}
}

static	void	activateCB_pushButton30(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget Viewports;
	
	XtPopdown(UxGetWidget(Viewports));
	}
}

static	void	activateCB_pushButton31(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget Viewports;
	
	ClearCycleButtons();
	
	if (gViewportActive) {
		ferret_command("CANCEL VIEWPORT", IGNORE_COMMAND_WIDGET);
		gViewportActive = False;
	}
	
	XtPopdown(UxGetWidget(Viewports));
	}
}

static	void	activateCB_pushButton32(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	JC_ClearButton_CB();
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_Viewports()
{
	Widget		_UxParent;


	/* Creation of Viewports */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	Viewports = XtVaCreatePopupShell( "Viewports",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 446,
			XmNheight, 154,
			XmNx, 270,
			XmNy, 337,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNiconName, "Ferret: Viewports",
			XmNtitle, "Ferret Viewports",
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( Viewports, XmNpopupCallback,
		(XtCallbackProc) popupCB_Viewports,
		(XtPointer) NULL );
	XtAddCallback( Viewports, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_Viewports,
		(XtPointer) NULL );


	createCB_Viewports( Viewports,
			(XtPointer) NULL, (XtPointer) NULL );


	/* Creation of form8 */
	form8 = XtVaCreateManagedWidget( "form8",
			xmFormWidgetClass,
			Viewports,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNwidth, 446,
			XmNheight, 154,
			NULL );


	/* Creation of frame14 */
	frame14 = XtVaCreateManagedWidget( "frame14",
			xmFrameWidgetClass,
			form8,
			XmNwidth, 441,
			XmNheight, 97,
			XmNx, 10,
			XmNy, 8,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of label69 */
	label69 = XtVaCreateManagedWidget( "label69",
			xmLabelWidgetClass,
			frame14,
			RES_CONVERT( XmNlabelString, "Predefined Viewports" ),
			XmNalignment, XmALIGNMENT_BEGINNING,
			XmNchildType, XmFRAME_TITLE_CHILD,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );


	/* Creation of form23 */
	form23 = XtVaCreateManagedWidget( "form23",
			xmFormWidgetClass,
			frame14,
			XmNresizePolicy, XmRESIZE_NONE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of rowColumn12 */
	rowColumn12 = XtVaCreateManagedWidget( "rowColumn12",
			xmRowColumnWidgetClass,
			form23,
			XmNnumColumns, 7,
			XmNorientation, XmHORIZONTAL,
			XmNradioBehavior, TRUE,
			XmNmarginHeight, 0,
			XmNradioAlwaysOne, TRUE,
			XmNwhichButton, 0,
			XmNmarginWidth, 2,
			XmNpacking, XmPACK_TIGHT,
			XmNspacing, 2,
			XmNentryAlignment, XmALIGNMENT_BEGINNING,
			XmNentryVerticalAlignment, XmALIGNMENT_BASELINE_BOTTOM,
			XmNisAligned, TRUE,
			XmNadjustMargin, TRUE,
			XmNresizeHeight, FALSE,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNtopOffset, 4,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			XmNheight, 35,
			NULL );


	/* Creation of toggleButton37 */
	toggleButton37 = XtVaCreateManagedWidget( "toggleButton37",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 2,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "Full" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNfillOnSelect, FALSE,
			XmNindicatorSize, 13,
			XmNset, TRUE,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton37, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton37,
		(XtPointer) NULL );
	XtAddCallback( toggleButton37, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton37,
		(XtPointer) NULL );



	/* Creation of toggleButton38 */
	toggleButton38 = XtVaCreateManagedWidget( "toggleButton38",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 40,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "LL" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNspacing, 1,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton38, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton38,
		(XtPointer) NULL );
	XtAddCallback( toggleButton38, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton38,
		(XtPointer) NULL );



	/* Creation of toggleButton39 */
	toggleButton39 = XtVaCreateManagedWidget( "toggleButton39",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 78,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "LR" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNspacing, 1,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton39, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton39,
		(XtPointer) NULL );
	XtAddCallback( toggleButton39, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton39,
		(XtPointer) NULL );



	/* Creation of toggleButton40 */
	toggleButton40 = XtVaCreateManagedWidget( "toggleButton40",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 116,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "UR" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNspacing, 1,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton40, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton40,
		(XtPointer) NULL );
	XtAddCallback( toggleButton40, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton40,
		(XtPointer) NULL );



	/* Creation of toggleButton41 */
	toggleButton41 = XtVaCreateManagedWidget( "toggleButton41",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 154,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "UL" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNspacing, 1,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton41, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton41,
		(XtPointer) NULL );
	XtAddCallback( toggleButton41, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton41,
		(XtPointer) NULL );



	/* Creation of toggleButton42 */
	toggleButton42 = XtVaCreateManagedWidget( "toggleButton42",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 192,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "Left" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton42, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton42,
		(XtPointer) NULL );
	XtAddCallback( toggleButton42, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton42,
		(XtPointer) NULL );



	/* Creation of toggleButton43 */
	toggleButton43 = XtVaCreateManagedWidget( "toggleButton43",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 230,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "Right" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton43, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton43,
		(XtPointer) NULL );
	XtAddCallback( toggleButton43, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton43,
		(XtPointer) NULL );



	/* Creation of toggleButton44 */
	toggleButton44 = XtVaCreateManagedWidget( "toggleButton44",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 268,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "Upper" ),
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton44, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton44,
		(XtPointer) NULL );
	XtAddCallback( toggleButton44, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton44,
		(XtPointer) NULL );



	/* Creation of toggleButton45 */
	toggleButton45 = XtVaCreateManagedWidget( "toggleButton45",
			xmToggleButtonWidgetClass,
			rowColumn12,
			XmNx, 306,
			XmNy, 0,
			XmNwidth, 36,
			XmNheight, 37,
			RES_CONVERT( XmNlabelString, "Lower" ),
			XmNindicatorOn, FALSE,
			RES_CONVERT( XmNhighlightColor, "#ff8f9e" ),
			XmNlabelType, XmPIXMAP,
			XmNmarginHeight, 0,
			XmNmarginLeft, 0,
			XmNmarginRight, 0,
			XmNmarginWidth, 0,
			XmNmarginBottom, 0,
			XmNmarginTop, 0,
			XmNindicatorSize, 13,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( toggleButton45, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton45,
		(XtPointer) NULL );
	XtAddCallback( toggleButton45, XmNarmCallback,
		(XtCallbackProc) armCB_toggleButton45,
		(XtPointer) NULL );



	/* Creation of toggleButton62 */
	toggleButton62 = XtVaCreateManagedWidget( "toggleButton62",
			xmToggleButtonWidgetClass,
			form23,
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNset, FALSE,
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNlabelString, "Cycle Corners" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 66,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn12,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );
	XtAddCallback( toggleButton62, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton62,
		(XtPointer) NULL );



	/* Creation of toggleButton63 */
	toggleButton63 = XtVaCreateManagedWidget( "toggleButton63",
			xmToggleButtonWidgetClass,
			form23,
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNset, FALSE,
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNlabelString, "Cy. L->R" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 29,
			XmNleftWidget, toggleButton62,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn12,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );
	XtAddCallback( toggleButton63, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton63,
		(XtPointer) NULL );



	/* Creation of toggleButton64 */
	toggleButton64 = XtVaCreateManagedWidget( "toggleButton64",
			xmToggleButtonWidgetClass,
			form23,
			XmNindicatorOn, FALSE,
			XmNlabelType, XmPIXMAP,
			XmNset, FALSE,
			XmNmarginBottom, 0,
			XmNmarginHeight, 0,
			XmNmarginTop, 0,
			XmNmarginWidth, 0,
			XmNspacing, 0,
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNlabelString, "Cy. U->D" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNleftOffset, 29,
			XmNleftWidget, toggleButton63,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNtopOffset, 1,
			XmNtopWidget, rowColumn12,
			XmNtopAttachment, XmATTACH_WIDGET,
			NULL );
	XtAddCallback( toggleButton64, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_toggleButton64,
		(XtPointer) NULL );



	/* Creation of rowColumn26 */
	rowColumn26 = XtVaCreateManagedWidget( "rowColumn26",
			xmRowColumnWidgetClass,
			form8,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_TIGHT,
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNtopOffset, 10,
			XmNtopWidget, frame14,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftPosition, 21,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of pushButton30 */
	pushButton30 = XtVaCreateManagedWidget( "pushButton30",
			xmPushButtonWidgetClass,
			rowColumn26,
			RES_CONVERT( XmNlabelString, "Done" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton30, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton30,
		(XtPointer) NULL );



	/* Creation of pushButton31 */
	pushButton31 = XtVaCreateManagedWidget( "pushButton31",
			xmPushButtonWidgetClass,
			rowColumn26,
			RES_CONVERT( XmNlabelString, "Cancel Viewports" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton31, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton31,
		(XtPointer) NULL );



	/* Creation of pushButton32 */
	pushButton32 = XtVaCreateManagedWidget( "pushButton32",
			xmPushButtonWidgetClass,
			rowColumn26,
			RES_CONVERT( XmNlabelString, "Clear" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton32, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton32,
		(XtPointer) NULL );




	return ( Viewports );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_Viewports( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedViewPorts == NULL) {
		rtrn = _Uxbuild_Viewports();

		/* install the pixmaps */
			InitPixmaps();
		}
		else
			rtrn = gSavedViewPorts;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		/* set the hi rez size */
		if (gHiRez && !gSavedViewPorts) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form8),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form8),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		if (!gSavedViewPorts)
			gSavedViewPorts = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

