
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

