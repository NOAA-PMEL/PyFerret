
/*******************************************************************************
	VectorOptions.c

       Associated Header file: VectorOptions.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/Scale.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "ferret_structures.h"

/* protos */
swidget create_VectorOptions(swidget UxParent);
static void UpdateStdLengthCB(void);
static void UpdateValStdLengthCB(void);

/* globals */
swidget VectorOptions;
swidget gSavedVectorOptions = NULL;
extern Boolean gHiRez;


static	Widget	form11;
static	Widget	scaleH2;
static	Widget	label9;
static	Widget	label54;
static	Widget	textField25;
static	Widget	rowColumn6;
static	Widget	pushButton20;
static	Widget	pushButton21;
static	Widget	pushButton22;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "VectorOptions.h"
#undef CONTEXT_MACRO_ACCESS

Widget	VectorOptions;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

static void UpdateStdLengthCB()
{
	float length;
	int val;
	short decPoints;

	return;
	/* get the value of the slider */
	XmScaleGetValue(UxGetWidget(scaleH2), &val);
	XtVaGetValues(UxGetWidget(scaleH2),
		XmNdecimalPoints, &decPoints,
		NULL);
	length = (float)val/pow(10.0, (double)decPoints);
	vectorLengthOptions.length = length;
}

static void UpdateValStdLengthCB()
{
	float val;
	char *tText;

	/* get the value of the text field */
	tText = (char *)XtMalloc(32);
	tText = XmTextFieldGetString(UxGetWidget(textField25));
	if (strlen(tText) == 0)
		vectorLengthOptions.value = UNSET_VALUE;
	else {	
		sscanf(tText, "%f", &val);
		vectorLengthOptions.value = val;
	}
	XtFree(tText); /* allocated by XtMalloc() */
}

static void DoOK()
{
	UpdateStdLengthCB();
	UpdateValStdLengthCB();
	XtPopdown(UxGetWidget(VectorOptions));
}

static void DoApply()
{
	UpdateStdLengthCB();
	UpdateValStdLengthCB();
}

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

static	void	destroyCB_VectorOptions(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	gSavedVectorOptions = NULL;
}

static	void	valueChangedCB_scaleH2(
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

static	void	valueChangedCB_textField25(
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

static	void	activateCB_pushButton20(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	extern swidget VectorOptions;
	
	XtPopdown(UxGetWidget(VectorOptions));
	}
}

static	void	activateCB_pushButton21(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	DoApply();
	}
}

static	void	activateCB_pushButton22(
			Widget wgt, 
			XtPointer cd, 
			XtPointer cb)
{
	Widget                  UxWidget = wgt;
	XtPointer               UxClientData = cd;
	XtPointer               UxCallbackArg = cb;
	{
	DoOK();
	}
}

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_VectorOptions()
{
	Widget		_UxParent;


	/* Creation of VectorOptions */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	VectorOptions = XtVaCreatePopupShell( "VectorOptions",
			topLevelShellWidgetClass,
			_UxParent,
			XmNwidth, 329,
			XmNheight, 123,
			XmNx, 300,
			XmNy, 545,
			XmNiconName, "Ferret: Vector Options",
			XmNtitle, "Ferret Vector Options",
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNallowShellResize, TRUE,
			NULL );
	XtAddCallback( VectorOptions, XmNdestroyCallback,
		(XtCallbackProc) destroyCB_VectorOptions,
		(XtPointer) NULL );



	/* Creation of form11 */
	form11 = XtVaCreateManagedWidget( "form11",
			xmFormWidgetClass,
			VectorOptions,
			XmNwidth, 329,
			XmNheight, 123,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, -3,
			XmNy, -2,
			XmNunitType, XmPIXELS,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );


	/* Creation of scaleH2 */
	scaleH2 = XtVaCreateManagedWidget( "scaleH2",
			xmScaleWidgetClass,
			form11,
			XmNwidth, 88,
			XmNheight, 41,
			XmNorientation, XmHORIZONTAL,
			XmNx, 175,
			XmNy, 36,
			XmNdecimalPoints, 2,
			XmNmaximum, 200,
			XmNshowValue, TRUE,
			RES_CONVERT( XmNtitleString, "" ),
			XmNfontList, UxConvertFontList( "-misc-fixed-medium-r-semicondensed--13-100-100-100-c-60-iso8859-1" ),
			XmNscaleHeight, 20,
			XmNvalue, 50,
			XmNmappedWhenManaged, FALSE,
			RES_CONVERT( XmNbackground, "gray80" ),
			NULL );
	XtAddCallback( scaleH2, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_scaleH2,
		(XtPointer) NULL );



	/* Creation of label9 */
	label9 = XtVaCreateManagedWidget( "label9",
			xmLabelWidgetClass,
			form11,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Standard Length (inches): 0.5" ),
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNtopOffset, 13,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of label54 */
	label54 = XtVaCreateManagedWidget( "label54",
			xmLabelWidgetClass,
			form11,
			XmNalignment, XmALIGNMENT_BEGINNING,
			RES_CONVERT( XmNlabelString, "Value of Standard Length:" ),
			XmNleftOffset, 15,
			XmNleftAttachment, XmATTACH_FORM,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNtopOffset, 45,
			XmNtopAttachment, XmATTACH_FORM,
			NULL );


	/* Creation of textField25 */
	textField25 = XtVaCreateManagedWidget( "textField25",
			xmTextFieldWidgetClass,
			form11,
			XmNsensitive, TRUE,
			RES_CONVERT( XmNbackground, "gray75" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-120-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			XmNtopOffset, 38,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 5,
			XmNleftWidget, label54,
			XmNleftAttachment, XmATTACH_WIDGET,
			XmNwidth, 95,
			XmNrightOffset, 10,
			XmNrightAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( textField25, XmNvalueChangedCallback,
		(XtCallbackProc) valueChangedCB_textField25,
		(XtPointer) NULL );



	/* Creation of rowColumn6 */
	rowColumn6 = XtVaCreateManagedWidget( "rowColumn6",
			xmRowColumnWidgetClass,
			form11,
			XmNentryAlignment, XmALIGNMENT_CENTER,
			XmNorientation, XmHORIZONTAL,
			XmNpacking, XmPACK_COLUMN,
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNtopOffset, 10,
			XmNtopWidget, textField25,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftPosition, 18,
			XmNleftOffset, 0,
			XmNleftAttachment, XmATTACH_POSITION,
			NULL );


	/* Creation of pushButton20 */
	pushButton20 = XtVaCreateManagedWidget( "pushButton20",
			xmPushButtonWidgetClass,
			rowColumn6,
			RES_CONVERT( XmNlabelString, "Cancel" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton20, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton20,
		(XtPointer) NULL );



	/* Creation of pushButton21 */
	pushButton21 = XtVaCreateManagedWidget( "pushButton21",
			xmPushButtonWidgetClass,
			rowColumn6,
			RES_CONVERT( XmNlabelString, "Apply" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton21, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton21,
		(XtPointer) NULL );



	/* Creation of pushButton22 */
	pushButton22 = XtVaCreateManagedWidget( "pushButton22",
			xmPushButtonWidgetClass,
			rowColumn6,
			RES_CONVERT( XmNlabelString, "OK" ),
			RES_CONVERT( XmNbackground, "gray80" ),
			XmNfontList, UxConvertFontList( "*courier-bold-r-*-140-*" ),
			RES_CONVERT( XmNforeground, "black" ),
			NULL );
	XtAddCallback( pushButton22, XmNactivateCallback,
		(XtCallbackProc) activateCB_pushButton22,
		(XtPointer) NULL );




	return ( VectorOptions );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_VectorOptions( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	{
		if (gSavedVectorOptions == NULL) {
		rtrn = _Uxbuild_VectorOptions();

		}
		else
			rtrn = gSavedVectorOptions;
		
		XtPopup(UxGetWidget(rtrn), no_grab);
		
		/* set the hi rez size */
		if (gHiRez && !gSavedVectorOptions) {
			Dimension width, height;
				
			XtVaGetValues(UxGetWidget(form11),
				XmNwidth, &width,
				XmNheight, &height,
				NULL);
			width = 1.2 * width;
			height = 1.1 * height;
				
			XtVaSetValues(UxGetWidget(form11),
				XmNwidth, width,
				XmNheight, height,
				NULL);
		}
		if (!gSavedVectorOptions)
			gSavedVectorOptions = rtrn;
		return(rtrn);
	}
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

