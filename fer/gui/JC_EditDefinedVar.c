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
	JC_EditDefinedVar.c

       Associated Header file: JC_EditDefinedVar.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/TextF.h>
#include <Xm/RowColumn.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "JC_EditDefinedVar_code.h"


static	Widget	form1;
static	Widget	Define_Button;
static	Widget	Cancel_Button;
static	Widget	EDV_rowColumn_definition;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "JC_EditDefinedVar.h"
#undef CONTEXT_MACRO_ACCESS

Widget	JC_EditDefinedVar;
Widget	EDV_rowColumn_Select;
Widget	EDV_textField_definition;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "JC_EditDefinedVar_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_JC_EditDefinedVar()
{
	Widget		_UxParent;


	/* Creation of JC_EditDefinedVar */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	JC_EditDefinedVar = XtVaCreatePopupShell( "JC_EditDefinedVar",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 450,
			XmNheight, 150,
			XmNx, 520,
			XmNy, 440,
			XmNiconName, "Ferret: Edit Defined Variable",
			XmNtitle, "Edit Defined Variable",
			NULL );


	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			JC_EditDefinedVar,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of Define_Button */
	Define_Button = XtVaCreateManagedWidget( "Define_Button",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			RES_CONVERT( XmNlabelString, "Define" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNleftOffset, 10,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( Define_Button, XmNactivateCallback,
		(XtCallbackProc) EDV_Define_CB,
		(XtPointer) NULL );



	/* Creation of Cancel_Button */
	Cancel_Button = XtVaCreateManagedWidget( "Cancel_Button",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "Cancel" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );
	XtAddCallback( Cancel_Button, XmNactivateCallback,
		(XtCallbackProc) JC_EDV_Cancel_CB,
		(XtPointer) NULL );



	/* Creation of EDV_rowColumn_Select */
	EDV_rowColumn_Select = XtVaCreateManagedWidget( "EDV_rowColumn_Select",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_FORM,
			XmNleftOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			NULL );


	/* Creation of EDV_rowColumn_definition */
	EDV_rowColumn_definition = XtVaCreateManagedWidget( "EDV_rowColumn_definition",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 120,
			XmNheight, 50,
			XmNx, 180,
			XmNy, 20,
			XmNleftAttachment, XmATTACH_FORM,
			XmNleftOffset, 90,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			NULL );


	/* Creation of EDV_textField_definition */
	EDV_textField_definition = XtVaCreateManagedWidget( "EDV_textField_definition",
			xmTextFieldWidgetClass,
			EDV_rowColumn_definition,
			NULL );



	return ( JC_EditDefinedVar );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_JC_EditDefinedVar( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_JC_EditDefinedVar();

	JC_EDV_Initialize();
	XtPopup(UxGetWidget(rtrn), no_grab);
	
	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

