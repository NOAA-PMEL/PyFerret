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
	JC_SelectRegridding.c

       Associated Header file: JC_SelectRegridding.h
*******************************************************************************/

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/Separator.h>
#include <Xm/TextF.h>
#include <Xm/RowColumn.h>
#include <Xm/Label.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "JC_SelectRegridding_code.h"


static	Widget	form1;
static	Widget	pushButton_Dismiss;
static	Widget	label1;
static	Widget	label_Grid;
static	Widget	form_uniform;
static	Widget	textField_G;
static	Widget	optionMenu_p1;
static	Widget	optionMenu_p1_LIN;
static	Widget	optionMenu_p1_AVE;
static	Widget	optionMenu_p1_ASN;
static	Widget	optionMenu_p1_separator1;
static	Widget	optionMenu_p1_VAR;
static	Widget	optionMenu_p1_NGD;
static	Widget	optionMenu_p1_SUM;
static	Widget	optionMenu_p1_MIN;
static	Widget	optionMenu_p1_MAX;
static	Widget	optionMenu_Transform_G;
static	Widget	form_non_uniform;
static	Widget	textField_GX;
static	Widget	textField_GY;
static	Widget	textField_GZ;
static	Widget	textField_GT;
static	Widget	optionMenu_p2;
static	Widget	optionMenu_GX_LIN;
static	Widget	optionMenu_GX_AVE;
static	Widget	optionMenu_GX_ASN;
static	Widget	optionMenu_GX_separator1;
static	Widget	optionMenu_GX_VAR;
static	Widget	optionMenu_GX_NGD;
static	Widget	optionMenu_GX_SUM;
static	Widget	optionMenu_GX_MIN;
static	Widget	optionMenu_GX_MAX;
static	Widget	optionMenu_TransformGX;
static	Widget	optionMenu_p3;
static	Widget	optionMenu_GY_LIN;
static	Widget	optionMenu_GY_AVE;
static	Widget	optionMenu_GY_ASN;
static	Widget	optionMenu_GY_separator1;
static	Widget	optionMenu_GY_VAR;
static	Widget	optionMenu_GY_NGD;
static	Widget	optionMenu_GY_SUM;
static	Widget	optionMenu_GY_MIN;
static	Widget	optionMenu_GY_MAX;
static	Widget	optionMenu_TransformGY;
static	Widget	optionMenu_p4;
static	Widget	optionMenu_GZ_LIN;
static	Widget	optionMenu_GZ_AVE;
static	Widget	optionMenu_GZ_ASN;
static	Widget	optionMenu_GZ_separator1;
static	Widget	optionMenu_GZ_VAR;
static	Widget	optionMenu_GZ_NGD;
static	Widget	optionMenu_GZ_SUM;
static	Widget	optionMenu_GZ_MIN;
static	Widget	optionMenu_GZ_MAX;
static	Widget	optionMenu_TransformGZ;
static	Widget	optionMenu_p5;
static	Widget	optionMenu_GT_LIN;
static	Widget	optionMenu_GT_AVE;
static	Widget	optionMenu_GT_ASN;
static	Widget	optionMenu_GT_separator1;
static	Widget	optionMenu_GT_VAR;
static	Widget	optionMenu_GT_NGD;
static	Widget	optionMenu_GT_SUM;
static	Widget	optionMenu_GT_MIN;
static	Widget	optionMenu_GT_MAX;
static	Widget	optionMenu_TransformGT;
static	Widget	label_GX;
static	Widget	label_GY;
static	Widget	label_GZ;
static	Widget	label_GT;
static	Widget	label_Transform;
static	Widget	pushButton_More;
static	Widget	pushButton_OK;
static	swidget	UxParent;

#define CONTEXT_MACRO_ACCESS 1
#include "JC_SelectRegridding.h"
#undef CONTEXT_MACRO_ACCESS

Widget	JC_SelectRegridding;
Widget	rowColumn_Select_G;
Widget	rowColumn_Select_GX;
Widget	rowColumn_Select_GY;
Widget	rowColumn_Select_GZ;
Widget	rowColumn_Select_GT;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "JC_SelectRegridding_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_JC_SelectRegridding()
{
	Widget		_UxParent;
	Widget		optionMenu_p1_shell;
	Widget		optionMenu_p2_shell;
	Widget		optionMenu_p3_shell;
	Widget		optionMenu_p4_shell;
	Widget		optionMenu_p5_shell;


	/* Creation of JC_SelectRegridding */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	JC_SelectRegridding = XtVaCreatePopupShell( "JC_SelectRegridding",
			topLevelShellWidgetClass,
			_UxParent,
			XmNiconName, "Ferret: Regridding",
			XmNtitle, "Ferret: Regridding",
			XmNallowShellResize, TRUE,
			XmNwidth, 475,
			NULL );


	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			JC_SelectRegridding,
			XmNwidth, 450,
			XmNheight, 160,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 50,
			XmNy, 60,
			XmNunitType, XmPIXELS,
			NULL );


	/* Creation of pushButton_Dismiss */
	pushButton_Dismiss = XtVaCreateManagedWidget( "pushButton_Dismiss",
			xmPushButtonWidgetClass,
			form1,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_NONE,
			RES_CONVERT( XmNlabelString, "Dismiss" ),
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNtopAttachment, XmATTACH_NONE,
			XmNwidth, 100,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );
	XtAddCallback( pushButton_Dismiss, XmNactivateCallback,
		(XtCallbackProc) JC_SR_DismissButton_CB,
		(XtPointer) NULL );



	/* Creation of label1 */
	label1 = XtVaCreateManagedWidget( "label1",
			xmLabelWidgetClass,
			form1,
			XmNleftOffset, 10,
			XmNtopAttachment, XmATTACH_FORM,
			XmNtopOffset, 10,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );


	/* Creation of label_Grid */
	label_Grid = XtVaCreateManagedWidget( "label_Grid",
			xmLabelWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Destination Grid" ),
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNtopWidget, label1,
			XmNleftOffset, 10,
			NULL );


	/* Creation of form_uniform */
	form_uniform = XtVaCreateManagedWidget( "form_uniform",
			xmFormWidgetClass,
			form1,
			XmNwidth, 400,
			XmNheight, 50,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNx, 0,
			XmNy, 20,
			XmNrightAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, label_Grid,
			XmNleftAttachment, XmATTACH_FORM,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, pushButton_Dismiss,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );


	/* Creation of rowColumn_Select_G */
	rowColumn_Select_G = XtVaCreateManagedWidget( "rowColumn_Select_G",
			xmRowColumnWidgetClass,
			form_uniform,
			XmNheight, 30,
			XmNy, 10,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 10,
			XmNtopAttachment, XmATTACH_POSITION,
			NULL );


	/* Creation of textField_G */
	textField_G = XtVaCreateManagedWidget( "textField_G",
			xmTextFieldWidgetClass,
			form_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNrightPosition, 70,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNeditable, FALSE,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopOffset, 4,
			NULL );


	/* Creation of optionMenu_p1 */
	optionMenu_p1_shell = XtVaCreatePopupShell ("optionMenu_p1_shell",
			xmMenuShellWidgetClass, form_uniform,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p1 = XtVaCreateWidget( "optionMenu_p1",
			xmRowColumnWidgetClass,
			optionMenu_p1_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_p1_LIN */
	optionMenu_p1_LIN = XtVaCreateManagedWidget( "optionMenu_p1_LIN",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "linear interp." ),
			NULL );
	XtAddCallback( optionMenu_p1_LIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "LIN" );



	/* Creation of optionMenu_p1_AVE */
	optionMenu_p1_AVE = XtVaCreateManagedWidget( "optionMenu_p1_AVE",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "weighted avg" ),
			NULL );
	XtAddCallback( optionMenu_p1_AVE, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "AVE" );



	/* Creation of optionMenu_p1_ASN */
	optionMenu_p1_ASN = XtVaCreateManagedWidget( "optionMenu_p1_ASN",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "by subscript" ),
			NULL );
	XtAddCallback( optionMenu_p1_ASN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "ASN" );



	/* Creation of optionMenu_p1_separator1 */
	optionMenu_p1_separator1 = XtVaCreateManagedWidget( "optionMenu_p1_separator1",
			xmSeparatorWidgetClass,
			optionMenu_p1,
			NULL );


	/* Creation of optionMenu_p1_VAR */
	optionMenu_p1_VAR = XtVaCreateManagedWidget( "optionMenu_p1_VAR",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "variance of pts" ),
			NULL );
	XtAddCallback( optionMenu_p1_VAR, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "VAR" );



	/* Creation of optionMenu_p1_NGD */
	optionMenu_p1_NGD = XtVaCreateManagedWidget( "optionMenu_p1_NGD",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "num source pts" ),
			NULL );
	XtAddCallback( optionMenu_p1_NGD, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "NGD" );



	/* Creation of optionMenu_p1_SUM */
	optionMenu_p1_SUM = XtVaCreateManagedWidget( "optionMenu_p1_SUM",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "weighted sum" ),
			NULL );
	XtAddCallback( optionMenu_p1_SUM, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "SUM" );



	/* Creation of optionMenu_p1_MIN */
	optionMenu_p1_MIN = XtVaCreateManagedWidget( "optionMenu_p1_MIN",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "minimum val" ),
			NULL );
	XtAddCallback( optionMenu_p1_MIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "MIN" );



	/* Creation of optionMenu_p1_MAX */
	optionMenu_p1_MAX = XtVaCreateManagedWidget( "optionMenu_p1_MAX",
			xmPushButtonWidgetClass,
			optionMenu_p1,
			RES_CONVERT( XmNlabelString, "maximum val" ),
			NULL );
	XtAddCallback( optionMenu_p1_MAX, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_G_CB,
		(XtPointer) "MAX" );



	/* Creation of optionMenu_Transform_G */
	optionMenu_Transform_G = XtVaCreateManagedWidget( "optionMenu_Transform_G",
			xmRowColumnWidgetClass,
			form_uniform,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p1,
			XmNy, 10,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			NULL );


	/* Creation of form_non_uniform */
	form_non_uniform = XtVaCreateManagedWidget( "form_non_uniform",
			xmFormWidgetClass,
			form1,
			XmNresizePolicy, XmRESIZE_NONE,
			XmNrightAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopWidget, label_Grid,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, pushButton_Dismiss,
			XmNbottomOffset, 10,
			XmNleftOffset, 2,
			XmNrightOffset, 2,
			NULL );


	/* Creation of rowColumn_Select_GX */
	rowColumn_Select_GX = XtVaCreateManagedWidget( "rowColumn_Select_GX",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNleftPosition, 10,
			NULL );


	/* Creation of rowColumn_Select_GY */
	rowColumn_Select_GY = XtVaCreateManagedWidget( "rowColumn_Select_GY",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 25,
			XmNleftPosition, 10,
			NULL );


	/* Creation of rowColumn_Select_GZ */
	rowColumn_Select_GZ = XtVaCreateManagedWidget( "rowColumn_Select_GZ",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 50,
			XmNleftPosition, 10,
			NULL );


	/* Creation of rowColumn_Select_GT */
	rowColumn_Select_GT = XtVaCreateManagedWidget( "rowColumn_Select_GT",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 75,
			XmNleftPosition, 10,
			NULL );


	/* Creation of textField_GX */
	textField_GX = XtVaCreateManagedWidget( "textField_GX",
			xmTextFieldWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNeditable, FALSE,
			XmNtopOffset, 6,
			NULL );


	/* Creation of textField_GY */
	textField_GY = XtVaCreateManagedWidget( "textField_GY",
			xmTextFieldWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 25,
			XmNeditable, FALSE,
			XmNtopOffset, 6,
			NULL );


	/* Creation of textField_GZ */
	textField_GZ = XtVaCreateManagedWidget( "textField_GZ",
			xmTextFieldWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 50,
			XmNeditable, FALSE,
			XmNtopOffset, 6,
			NULL );


	/* Creation of textField_GT */
	textField_GT = XtVaCreateManagedWidget( "textField_GT",
			xmTextFieldWidgetClass,
			form_non_uniform,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 30,
			XmNrightAttachment, XmATTACH_POSITION,
			XmNrightPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 75,
			XmNeditable, FALSE,
			XmNtopOffset, 6,
			NULL );


	/* Creation of optionMenu_p2 */
	optionMenu_p2_shell = XtVaCreatePopupShell ("optionMenu_p2_shell",
			xmMenuShellWidgetClass, form_non_uniform,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p2 = XtVaCreateWidget( "optionMenu_p2",
			xmRowColumnWidgetClass,
			optionMenu_p2_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_GX_LIN */
	optionMenu_GX_LIN = XtVaCreateManagedWidget( "optionMenu_GX_LIN",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "linear interp." ),
			NULL );
	XtAddCallback( optionMenu_GX_LIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "LIN" );



	/* Creation of optionMenu_GX_AVE */
	optionMenu_GX_AVE = XtVaCreateManagedWidget( "optionMenu_GX_AVE",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "weighted avg" ),
			NULL );
	XtAddCallback( optionMenu_GX_AVE, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "AVE" );



	/* Creation of optionMenu_GX_ASN */
	optionMenu_GX_ASN = XtVaCreateManagedWidget( "optionMenu_GX_ASN",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "by subscript" ),
			NULL );
	XtAddCallback( optionMenu_GX_ASN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "ASN" );



	/* Creation of optionMenu_GX_separator1 */
	optionMenu_GX_separator1 = XtVaCreateManagedWidget( "optionMenu_GX_separator1",
			xmSeparatorWidgetClass,
			optionMenu_p2,
			NULL );


	/* Creation of optionMenu_GX_VAR */
	optionMenu_GX_VAR = XtVaCreateManagedWidget( "optionMenu_GX_VAR",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "variance of pts" ),
			NULL );
	XtAddCallback( optionMenu_GX_VAR, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "VAR" );



	/* Creation of optionMenu_GX_NGD */
	optionMenu_GX_NGD = XtVaCreateManagedWidget( "optionMenu_GX_NGD",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "num source pts" ),
			NULL );
	XtAddCallback( optionMenu_GX_NGD, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "NGD" );



	/* Creation of optionMenu_GX_SUM */
	optionMenu_GX_SUM = XtVaCreateManagedWidget( "optionMenu_GX_SUM",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "weighted sum" ),
			NULL );
	XtAddCallback( optionMenu_GX_SUM, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "SUM" );



	/* Creation of optionMenu_GX_MIN */
	optionMenu_GX_MIN = XtVaCreateManagedWidget( "optionMenu_GX_MIN",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "minimum val" ),
			NULL );
	XtAddCallback( optionMenu_GX_MIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "MIN" );



	/* Creation of optionMenu_GX_MAX */
	optionMenu_GX_MAX = XtVaCreateManagedWidget( "optionMenu_GX_MAX",
			xmPushButtonWidgetClass,
			optionMenu_p2,
			RES_CONVERT( XmNlabelString, "maximum val" ),
			NULL );
	XtAddCallback( optionMenu_GX_MAX, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GX_CB,
		(XtPointer) "MAX" );



	/* Creation of optionMenu_TransformGX */
	optionMenu_TransformGX = XtVaCreateManagedWidget( "optionMenu_TransformGX",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p2,
			XmNx, 190,
			XmNy, 150,
			XmNwidth, 100,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			NULL );


	/* Creation of optionMenu_p3 */
	optionMenu_p3_shell = XtVaCreatePopupShell ("optionMenu_p3_shell",
			xmMenuShellWidgetClass, form_non_uniform,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p3 = XtVaCreateWidget( "optionMenu_p3",
			xmRowColumnWidgetClass,
			optionMenu_p3_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_GY_LIN */
	optionMenu_GY_LIN = XtVaCreateManagedWidget( "optionMenu_GY_LIN",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "linear interp." ),
			NULL );
	XtAddCallback( optionMenu_GY_LIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "LIN" );



	/* Creation of optionMenu_GY_AVE */
	optionMenu_GY_AVE = XtVaCreateManagedWidget( "optionMenu_GY_AVE",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "weighted avg" ),
			NULL );
	XtAddCallback( optionMenu_GY_AVE, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "AVE" );



	/* Creation of optionMenu_GY_ASN */
	optionMenu_GY_ASN = XtVaCreateManagedWidget( "optionMenu_GY_ASN",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "by subscript" ),
			NULL );
	XtAddCallback( optionMenu_GY_ASN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "ASN" );



	/* Creation of optionMenu_GY_separator1 */
	optionMenu_GY_separator1 = XtVaCreateManagedWidget( "optionMenu_GY_separator1",
			xmSeparatorWidgetClass,
			optionMenu_p3,
			NULL );


	/* Creation of optionMenu_GY_VAR */
	optionMenu_GY_VAR = XtVaCreateManagedWidget( "optionMenu_GY_VAR",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "variance of pts" ),
			NULL );
	XtAddCallback( optionMenu_GY_VAR, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "VAR" );



	/* Creation of optionMenu_GY_NGD */
	optionMenu_GY_NGD = XtVaCreateManagedWidget( "optionMenu_GY_NGD",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "num source pts" ),
			NULL );
	XtAddCallback( optionMenu_GY_NGD, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "NGD" );



	/* Creation of optionMenu_GY_SUM */
	optionMenu_GY_SUM = XtVaCreateManagedWidget( "optionMenu_GY_SUM",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "weighted sum" ),
			NULL );
	XtAddCallback( optionMenu_GY_SUM, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "SUM" );



	/* Creation of optionMenu_GY_MIN */
	optionMenu_GY_MIN = XtVaCreateManagedWidget( "optionMenu_GY_MIN",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "minimum val" ),
			NULL );
	XtAddCallback( optionMenu_GY_MIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "MIN" );



	/* Creation of optionMenu_GY_MAX */
	optionMenu_GY_MAX = XtVaCreateManagedWidget( "optionMenu_GY_MAX",
			xmPushButtonWidgetClass,
			optionMenu_p3,
			RES_CONVERT( XmNlabelString, "maximum val" ),
			NULL );
	XtAddCallback( optionMenu_GY_MAX, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GY_CB,
		(XtPointer) "MAX" );



	/* Creation of optionMenu_TransformGY */
	optionMenu_TransformGY = XtVaCreateManagedWidget( "optionMenu_TransformGY",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p3,
			XmNx, 190,
			XmNy, 180,
			XmNwidth, 100,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 25,
			NULL );


	/* Creation of optionMenu_p4 */
	optionMenu_p4_shell = XtVaCreatePopupShell ("optionMenu_p4_shell",
			xmMenuShellWidgetClass, form_non_uniform,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p4 = XtVaCreateWidget( "optionMenu_p4",
			xmRowColumnWidgetClass,
			optionMenu_p4_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_GZ_LIN */
	optionMenu_GZ_LIN = XtVaCreateManagedWidget( "optionMenu_GZ_LIN",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "linear interp." ),
			NULL );
	XtAddCallback( optionMenu_GZ_LIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "LIN" );



	/* Creation of optionMenu_GZ_AVE */
	optionMenu_GZ_AVE = XtVaCreateManagedWidget( "optionMenu_GZ_AVE",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "weighted avg" ),
			NULL );
	XtAddCallback( optionMenu_GZ_AVE, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "AVE" );



	/* Creation of optionMenu_GZ_ASN */
	optionMenu_GZ_ASN = XtVaCreateManagedWidget( "optionMenu_GZ_ASN",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "by subscript" ),
			NULL );
	XtAddCallback( optionMenu_GZ_ASN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "ASN" );



	/* Creation of optionMenu_GZ_separator1 */
	optionMenu_GZ_separator1 = XtVaCreateManagedWidget( "optionMenu_GZ_separator1",
			xmSeparatorWidgetClass,
			optionMenu_p4,
			NULL );


	/* Creation of optionMenu_GZ_VAR */
	optionMenu_GZ_VAR = XtVaCreateManagedWidget( "optionMenu_GZ_VAR",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "variance of pts" ),
			NULL );
	XtAddCallback( optionMenu_GZ_VAR, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "VAR" );



	/* Creation of optionMenu_GZ_NGD */
	optionMenu_GZ_NGD = XtVaCreateManagedWidget( "optionMenu_GZ_NGD",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "num source pts" ),
			NULL );
	XtAddCallback( optionMenu_GZ_NGD, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "NGD" );



	/* Creation of optionMenu_GZ_SUM */
	optionMenu_GZ_SUM = XtVaCreateManagedWidget( "optionMenu_GZ_SUM",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "weighted sum" ),
			NULL );
	XtAddCallback( optionMenu_GZ_SUM, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "SUM" );



	/* Creation of optionMenu_GZ_MIN */
	optionMenu_GZ_MIN = XtVaCreateManagedWidget( "optionMenu_GZ_MIN",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "minimum val" ),
			NULL );
	XtAddCallback( optionMenu_GZ_MIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "MIN" );



	/* Creation of optionMenu_GZ_MAX */
	optionMenu_GZ_MAX = XtVaCreateManagedWidget( "optionMenu_GZ_MAX",
			xmPushButtonWidgetClass,
			optionMenu_p4,
			RES_CONVERT( XmNlabelString, "maximum val" ),
			NULL );
	XtAddCallback( optionMenu_GZ_MAX, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GZ_CB,
		(XtPointer) "MAX" );



	/* Creation of optionMenu_TransformGZ */
	optionMenu_TransformGZ = XtVaCreateManagedWidget( "optionMenu_TransformGZ",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p4,
			XmNx, 190,
			XmNy, 220,
			XmNwidth, 100,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 50,
			NULL );


	/* Creation of optionMenu_p5 */
	optionMenu_p5_shell = XtVaCreatePopupShell ("optionMenu_p5_shell",
			xmMenuShellWidgetClass, form_non_uniform,
			XmNwidth, 1,
			XmNheight, 1,
			XmNallowShellResize, TRUE,
			XmNoverrideRedirect, TRUE,
			NULL );

	optionMenu_p5 = XtVaCreateWidget( "optionMenu_p5",
			xmRowColumnWidgetClass,
			optionMenu_p5_shell,
			XmNrowColumnType, XmMENU_PULLDOWN,
			NULL );


	/* Creation of optionMenu_GT_LIN */
	optionMenu_GT_LIN = XtVaCreateManagedWidget( "optionMenu_GT_LIN",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "linear interp." ),
			NULL );
	XtAddCallback( optionMenu_GT_LIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "LIN" );



	/* Creation of optionMenu_GT_AVE */
	optionMenu_GT_AVE = XtVaCreateManagedWidget( "optionMenu_GT_AVE",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "weighted avg" ),
			NULL );
	XtAddCallback( optionMenu_GT_AVE, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "AVE" );



	/* Creation of optionMenu_GT_ASN */
	optionMenu_GT_ASN = XtVaCreateManagedWidget( "optionMenu_GT_ASN",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "by subscript" ),
			NULL );
	XtAddCallback( optionMenu_GT_ASN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "ASN" );



	/* Creation of optionMenu_GT_separator1 */
	optionMenu_GT_separator1 = XtVaCreateManagedWidget( "optionMenu_GT_separator1",
			xmSeparatorWidgetClass,
			optionMenu_p5,
			NULL );


	/* Creation of optionMenu_GT_VAR */
	optionMenu_GT_VAR = XtVaCreateManagedWidget( "optionMenu_GT_VAR",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "variance of pts" ),
			NULL );
	XtAddCallback( optionMenu_GT_VAR, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "VAR" );



	/* Creation of optionMenu_GT_NGD */
	optionMenu_GT_NGD = XtVaCreateManagedWidget( "optionMenu_GT_NGD",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "num source pts" ),
			NULL );
	XtAddCallback( optionMenu_GT_NGD, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "NGD" );



	/* Creation of optionMenu_GT_SUM */
	optionMenu_GT_SUM = XtVaCreateManagedWidget( "optionMenu_GT_SUM",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "weighted sum" ),
			NULL );
	XtAddCallback( optionMenu_GT_SUM, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "SUM" );



	/* Creation of optionMenu_GT_MIN */
	optionMenu_GT_MIN = XtVaCreateManagedWidget( "optionMenu_GT_MIN",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "minimum val" ),
			NULL );
	XtAddCallback( optionMenu_GT_MIN, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "MIN" );



	/* Creation of optionMenu_GT_MAX */
	optionMenu_GT_MAX = XtVaCreateManagedWidget( "optionMenu_GT_MAX",
			xmPushButtonWidgetClass,
			optionMenu_p5,
			RES_CONVERT( XmNlabelString, "maximum val" ),
			NULL );
	XtAddCallback( optionMenu_GT_MAX, XmNactivateCallback,
		(XtCallbackProc) JC_SR_TransMenu_GT_CB,
		(XtPointer) "MAX" );



	/* Creation of optionMenu_TransformGT */
	optionMenu_TransformGT = XtVaCreateManagedWidget( "optionMenu_TransformGT",
			xmRowColumnWidgetClass,
			form_non_uniform,
			XmNrowColumnType, XmMENU_OPTION,
			XmNsubMenuId, optionMenu_p5,
			XmNx, 190,
			XmNy, 120,
			XmNwidth, 100,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftPosition, 70,
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 75,
			NULL );


	/* Creation of label_GX */
	label_GX = XtVaCreateManagedWidget( "label_GX",
			xmLabelWidgetClass,
			form_non_uniform,
			RES_CONVERT( XmNlabelString, "X" ),
			XmNtopAttachment, XmATTACH_POSITION,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 17,
			XmNtopOffset, 10,
			NULL );


	/* Creation of label_GY */
	label_GY = XtVaCreateManagedWidget( "label_GY",
			xmLabelWidgetClass,
			form_non_uniform,
			RES_CONVERT( XmNlabelString, "Y" ),
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 25,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 17,
			XmNtopOffset, 10,
			NULL );


	/* Creation of label_GZ */
	label_GZ = XtVaCreateManagedWidget( "label_GZ",
			xmLabelWidgetClass,
			form_non_uniform,
			RES_CONVERT( XmNlabelString, "Z" ),
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 50,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 17,
			XmNtopOffset, 10,
			NULL );


	/* Creation of label_GT */
	label_GT = XtVaCreateManagedWidget( "label_GT",
			xmLabelWidgetClass,
			form_non_uniform,
			RES_CONVERT( XmNlabelString, "T" ),
			XmNtopAttachment, XmATTACH_POSITION,
			XmNtopPosition, 75,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 17,
			XmNtopOffset, 10,
			NULL );


	/* Creation of label_Transform */
	label_Transform = XtVaCreateManagedWidget( "label_Transform",
			xmLabelWidgetClass,
			form1,
			RES_CONVERT( XmNlabelString, "Method" ),
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopOffset, 10,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNleftPosition, 70,
			XmNtopWidget, label1,
			XmNleftOffset, 17,
			NULL );


	/* Creation of pushButton_More */
	pushButton_More = XtVaCreateManagedWidget( "pushButton_More",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "More..." ),
			XmNrecomputeSize, FALSE,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, -50,
			XmNleftPosition, 50,
			NULL );
	XtAddCallback( pushButton_More, XmNactivateCallback,
		(XtCallbackProc) JC_SR_MoreButton_CB,
		(XtPointer) NULL );



	/* Creation of pushButton_OK */
	pushButton_OK = XtVaCreateManagedWidget( "pushButton_OK",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			XmNheight, 30,
			RES_CONVERT( XmNlabelString, "OK" ),
			XmNrecomputeSize, FALSE,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 10,
			XmNleftPosition, 0,
			NULL );
	XtAddCallback( pushButton_OK, XmNactivateCallback,
		(XtCallbackProc) JC_SR_OKButton_CB,
		(XtPointer) NULL );




	return ( JC_SelectRegridding );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_JC_SelectRegridding( swidget _UxUxParent )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;

	rtrn = _Uxbuild_JC_SelectRegridding();

	JC_SR_Initialize();
	XtPopup(UxGetWidget(rtrn), no_grab);
	
	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

