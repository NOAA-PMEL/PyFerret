
/*******************************************************************************
	JC_DefineVariable.c

       Associated Header file: JC_DefineVariable.h
*******************************************************************************/

#include <stdio.h>
#include <Xm/Xm.h>
#include <Xm/MwmUtil.h>
#include <Xm/MenuShell.h>
#include "UxXt.h"

#include <Xm/TextF.h>
#include <Xm/Label.h>
#include <Xm/RowColumn.h>
#include <Xm/PushB.h>
#include <Xm/Form.h>
#include <X11/Shell.h>

/*******************************************************************************
       Includes, Defines, and Global variables from the Declarations Editor:
*******************************************************************************/

#include "JC_DefineVariable_code.h"


static	Widget	form1;
static	Widget	pushButton1;
static	Widget	pushButton3;
static	Widget	rowColumn1;
static	Widget	label_Title;
static	Widget	textField_Title;
static	Widget	label_Units;
static	Widget	textField_Units;
static	Widget	rowColumn_F;
static	Widget	rowColumn1_F;
static	Widget	label_F_LET;
static	Widget	textField_F_VarName;
static	Widget	label_F_Equals;
static	Widget	rowColumn_var1_F;
static	Widget	label_LeftParen_1var;
static	Widget	textField1_1var;
static	Widget	label_Times_1var;
static	Widget	textField2_1var2;
static	Widget	rowColumn_var2_F;
static	Widget	label1_var2;
static	Widget	textField1_var2;
static	Widget	label2_var1;
static	Widget	textField2_var2;
static	Widget	rowColumn_var3_F;
static	Widget	label1_var3;
static	Widget	textField1_var3;
static	Widget	label2_var3;
static	Widget	textField2_var3;
static	Widget	rowColumn_end_F;
static	Widget	label1_end;
static	Widget	rowColumn_EXP;
static	Widget	rowColumn1_EXP;
static	Widget	label1_EXP;
static	Widget	textField1_EXP;
static	Widget	label2_EXP;
static	Widget	label3_EXP;
static	Widget	textField2_EXP;
static	Widget	label4_EXP;
static	Widget	textField3_EXP;
static	Widget	label5_EXP;
static	Widget	rowColumn2_EXP;
static	Widget	textField4_EXP;
static	Widget	rowColumn_LC;
static	Widget	rowColumn1_LC;
static	Widget	label_LC_LET;
static	Widget	textField_LC_VarName;
static	Widget	label_LC_Equals;
static	Widget	textField_LC_1;
static	Widget	label_LC_Times1;
static	Widget	textField_LC_2;
static	Widget	labe5_LC;
static	Widget	rowColumn2_LC;
static	Widget	textField_LC_3;
static	Widget	label_LC_Times2;
static	Widget	textField_LC_4;
static	Widget	label6_LC;
static	Widget	label7_LC;
static	Widget	label_dset_var1;
static	Widget	label_dset_var2;
static	Widget	label_dset_var3;
static	Widget	label_dset1_LC;
static	Widget	label_dset2_LC;
static	Widget	label_dset1_EXP;
static	swidget	UxParent;
static	int	functional_form;

#define CONTEXT_MACRO_ACCESS 1
#include "JC_DefineVariable.h"
#undef CONTEXT_MACRO_ACCESS

Widget	JC_DefineVariable;
Widget	rowColumn_Select2_LC;
Widget	rowColumn_Select1_LC;
Widget	rowColumn_Select1_EXP;
Widget	rowColumn_Select1_F;
Widget	rowColumn_Select_var2;
Widget	rowColumn_Select_var3;

/*******************************************************************************
Auxiliary code from the Declarations Editor:
*******************************************************************************/

#include "JC_DefineVariable_code.c"

/*******************************************************************************
       The following are callback functions.
*******************************************************************************/

/*******************************************************************************
       The 'build_' function creates all the widgets
       using the resource values specified in the Property Editor.
*******************************************************************************/

static Widget	_Uxbuild_JC_DefineVariable()
{
	Widget		_UxParent;


	/* Creation of JC_DefineVariable */
	_UxParent = UxParent;
	if ( _UxParent == NULL )
	{
		_UxParent = UxTopLevel;
	}

	JC_DefineVariable = XtVaCreatePopupShell( "JC_DefineVariable",
			topLevelShellWidgetClass,
			_UxParent,
			XmNallowShellResize, TRUE,
			XmNminWidth, 680,
			XmNx, 200,
			XmNy, 400,
			XmNiconName, "Ferret: Define Variable",
			XmNtitle, "Define Variable",
			NULL );


	/* Creation of form1 */
	form1 = XtVaCreateManagedWidget( "form1",
			xmFormWidgetClass,
			JC_DefineVariable,
			XmNnoResize, TRUE,
			NULL );


	/* Creation of pushButton1 */
	pushButton1 = XtVaCreateManagedWidget( "pushButton1",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			XmNheight, 30,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			RES_CONVERT( XmNlabelString, "OK" ),
			XmNleftOffset, 10,
			XmNleftAttachment, XmATTACH_FORM,
			NULL );
	XtAddCallback( pushButton1, XmNactivateCallback,
		(XtCallbackProc) JC_DV_Button1_CB,
		(XtPointer) NULL );



	/* Creation of pushButton3 */
	pushButton3 = XtVaCreateManagedWidget( "pushButton3",
			xmPushButtonWidgetClass,
			form1,
			XmNwidth, 100,
			XmNheight, 30,
			XmNbottomAttachment, XmATTACH_FORM,
			XmNbottomOffset, 10,
			RES_CONVERT( XmNlabelString, "Cancel" ),
			XmNleftAttachment, XmATTACH_NONE,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 10,
			NULL );
	XtAddCallback( pushButton3, XmNactivateCallback,
		(XtCallbackProc) JC_DV_Button3_CB,
		(XtPointer) NULL );



	/* Creation of rowColumn1 */
	rowColumn1 = XtVaCreateWidget( "rowColumn1",
			xmRowColumnWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, pushButton1,
			XmNleftAttachment, XmATTACH_FORM,
			XmNorientation, XmHORIZONTAL,
			XmNleftOffset, 5,
			XmNbottomOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			NULL );


	/* Creation of label_Title */
	label_Title = XtVaCreateWidget( "label_Title",
			xmLabelWidgetClass,
			rowColumn1,
			RES_CONVERT( XmNlabelString, "Title:" ),
			NULL );


	/* Creation of textField_Title */
	textField_Title = XtVaCreateWidget( "textField_Title",
			xmTextFieldWidgetClass,
			rowColumn1,
			NULL );


	/* Creation of label_Units */
	label_Units = XtVaCreateWidget( "label_Units",
			xmLabelWidgetClass,
			rowColumn1,
			RES_CONVERT( XmNlabelString, "Units:" ),
			NULL );


	/* Creation of textField_Units */
	textField_Units = XtVaCreateWidget( "textField_Units",
			xmTextFieldWidgetClass,
			rowColumn1,
			NULL );


	/* Creation of rowColumn_F */
	rowColumn_F = XtVaCreateWidget( "rowColumn_F",
			xmRowColumnWidgetClass,
			form1,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_FORM,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn1,
			XmNleftOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNbottomOffset, 40,
			XmNtopOffset, 30,
			NULL );


	/* Creation of rowColumn1_F */
	rowColumn1_F = XtVaCreateWidget( "rowColumn1_F",
			xmRowColumnWidgetClass,
			rowColumn_F,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label_F_LET */
	label_F_LET = XtVaCreateManagedWidget( "label_F_LET",
			xmLabelWidgetClass,
			rowColumn1_F,
			RES_CONVERT( XmNlabelString, "LET" ),
			NULL );


	/* Creation of textField_F_VarName */
	textField_F_VarName = XtVaCreateManagedWidget( "textField_F_VarName",
			xmTextFieldWidgetClass,
			rowColumn1_F,
			NULL );


	/* Creation of label_F_Equals */
	label_F_Equals = XtVaCreateManagedWidget( "label_F_Equals",
			xmLabelWidgetClass,
			rowColumn1_F,
			RES_CONVERT( XmNlabelString, "=" ),
			NULL );


	/* Creation of rowColumn_var1_F */
	rowColumn_var1_F = XtVaCreateWidget( "rowColumn_var1_F",
			xmRowColumnWidgetClass,
			rowColumn_F,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label_LeftParen_1var */
	label_LeftParen_1var = XtVaCreateManagedWidget( "label_LeftParen_1var",
			xmLabelWidgetClass,
			rowColumn_var1_F,
			RES_CONVERT( XmNlabelString, "(" ),
			NULL );


	/* Creation of textField1_1var */
	textField1_1var = XtVaCreateManagedWidget( "textField1_1var",
			xmTextFieldWidgetClass,
			rowColumn_var1_F,
			XmNcolumns, 5,
			NULL );


	/* Creation of label_Times_1var */
	label_Times_1var = XtVaCreateManagedWidget( "label_Times_1var",
			xmLabelWidgetClass,
			rowColumn_var1_F,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField2_1var2 */
	textField2_1var2 = XtVaCreateManagedWidget( "textField2_1var2",
			xmTextFieldWidgetClass,
			rowColumn_var1_F,
			XmNcolumns, 10,
			NULL );


	/* Creation of rowColumn_var2_F */
	rowColumn_var2_F = XtVaCreateWidget( "rowColumn_var2_F",
			xmRowColumnWidgetClass,
			rowColumn_F,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label1_var2 */
	label1_var2 = XtVaCreateManagedWidget( "label1_var2",
			xmLabelWidgetClass,
			rowColumn_var2_F,
			RES_CONVERT( XmNlabelString, "," ),
			NULL );


	/* Creation of textField1_var2 */
	textField1_var2 = XtVaCreateManagedWidget( "textField1_var2",
			xmTextFieldWidgetClass,
			rowColumn_var2_F,
			XmNcolumns, 5,
			NULL );


	/* Creation of label2_var1 */
	label2_var1 = XtVaCreateManagedWidget( "label2_var1",
			xmLabelWidgetClass,
			rowColumn_var2_F,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField2_var2 */
	textField2_var2 = XtVaCreateManagedWidget( "textField2_var2",
			xmTextFieldWidgetClass,
			rowColumn_var2_F,
			XmNcolumns, 10,
			NULL );


	/* Creation of rowColumn_var3_F */
	rowColumn_var3_F = XtVaCreateWidget( "rowColumn_var3_F",
			xmRowColumnWidgetClass,
			rowColumn_F,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label1_var3 */
	label1_var3 = XtVaCreateManagedWidget( "label1_var3",
			xmLabelWidgetClass,
			rowColumn_var3_F,
			RES_CONVERT( XmNlabelString, "," ),
			NULL );


	/* Creation of textField1_var3 */
	textField1_var3 = XtVaCreateManagedWidget( "textField1_var3",
			xmTextFieldWidgetClass,
			rowColumn_var3_F,
			NULL );


	/* Creation of label2_var3 */
	label2_var3 = XtVaCreateManagedWidget( "label2_var3",
			xmLabelWidgetClass,
			rowColumn_var3_F,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField2_var3 */
	textField2_var3 = XtVaCreateManagedWidget( "textField2_var3",
			xmTextFieldWidgetClass,
			rowColumn_var3_F,
			NULL );


	/* Creation of rowColumn_end_F */
	rowColumn_end_F = XtVaCreateWidget( "rowColumn_end_F",
			xmRowColumnWidgetClass,
			rowColumn_F,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label1_end */
	label1_end = XtVaCreateManagedWidget( "label1_end",
			xmLabelWidgetClass,
			rowColumn_end_F,
			RES_CONVERT( XmNlabelString, ")" ),
			NULL );


	/* Creation of rowColumn_EXP */
	rowColumn_EXP = XtVaCreateWidget( "rowColumn_EXP",
			xmRowColumnWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn1,
			XmNleftAttachment, XmATTACH_FORM,
			XmNorientation, XmHORIZONTAL,
			XmNtopAttachment, XmATTACH_FORM,
			XmNleftOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNbottomOffset, 40,
			XmNtopOffset, 30,
			NULL );


	/* Creation of rowColumn1_EXP */
	rowColumn1_EXP = XtVaCreateWidget( "rowColumn1_EXP",
			xmRowColumnWidgetClass,
			rowColumn_EXP,
			XmNwidth, 60,
			XmNheight, 10,
			XmNx, 140,
			XmNy, 10,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label1_EXP */
	label1_EXP = XtVaCreateManagedWidget( "label1_EXP",
			xmLabelWidgetClass,
			rowColumn1_EXP,
			RES_CONVERT( XmNlabelString, "LET" ),
			NULL );


	/* Creation of textField1_EXP */
	textField1_EXP = XtVaCreateManagedWidget( "textField1_EXP",
			xmTextFieldWidgetClass,
			rowColumn1_EXP,
			XmNwidth, 20,
			XmNx, 10,
			XmNy, 10,
			XmNheight, 10,
			NULL );


	/* Creation of label2_EXP */
	label2_EXP = XtVaCreateManagedWidget( "label2_EXP",
			xmLabelWidgetClass,
			rowColumn1_EXP,
			RES_CONVERT( XmNlabelString, "=" ),
			NULL );


	/* Creation of label3_EXP */
	label3_EXP = XtVaCreateManagedWidget( "label3_EXP",
			xmLabelWidgetClass,
			rowColumn1_EXP,
			RES_CONVERT( XmNlabelString, "(" ),
			NULL );


	/* Creation of textField2_EXP */
	textField2_EXP = XtVaCreateManagedWidget( "textField2_EXP",
			xmTextFieldWidgetClass,
			rowColumn1_EXP,
			XmNwidth, 20,
			XmNx, 110,
			XmNy, 10,
			XmNheight, 10,
			XmNcolumns, 5,
			NULL );


	/* Creation of label4_EXP */
	label4_EXP = XtVaCreateManagedWidget( "label4_EXP",
			xmLabelWidgetClass,
			rowColumn1_EXP,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField3_EXP */
	textField3_EXP = XtVaCreateManagedWidget( "textField3_EXP",
			xmTextFieldWidgetClass,
			rowColumn1_EXP,
			XmNwidth, 60,
			XmNx, 190,
			XmNy, 10,
			XmNheight, 10,
			XmNcolumns, 10,
			NULL );


	/* Creation of label5_EXP */
	label5_EXP = XtVaCreateManagedWidget( "label5_EXP",
			xmLabelWidgetClass,
			rowColumn1_EXP,
			RES_CONVERT( XmNlabelString, ")" ),
			NULL );


	/* Creation of rowColumn2_EXP */
	rowColumn2_EXP = XtVaCreateWidget( "rowColumn2_EXP",
			xmRowColumnWidgetClass,
			rowColumn_EXP,
			XmNwidth, 90,
			XmNheight, 30,
			XmNx, 170,
			XmNy, 0,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField4_EXP */
	textField4_EXP = XtVaCreateManagedWidget( "textField4_EXP",
			xmTextFieldWidgetClass,
			rowColumn2_EXP,
			XmNwidth, 40,
			XmNx, 260,
			XmNy, 10,
			XmNheight, 10,
			XmNcolumns, 5,
			NULL );


	/* Creation of rowColumn_LC */
	rowColumn_LC = XtVaCreateWidget( "rowColumn_LC",
			xmRowColumnWidgetClass,
			form1,
			XmNorientation, XmHORIZONTAL,
			XmNleftAttachment, XmATTACH_FORM,
			XmNtopAttachment, XmATTACH_FORM,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn1,
			XmNleftOffset, 5,
			XmNrightAttachment, XmATTACH_FORM,
			XmNrightOffset, 5,
			XmNbottomOffset, 40,
			XmNtopOffset, 30,
			NULL );


	/* Creation of rowColumn1_LC */
	rowColumn1_LC = XtVaCreateWidget( "rowColumn1_LC",
			xmRowColumnWidgetClass,
			rowColumn_LC,
			XmNwidth, 30,
			XmNheight, 20,
			XmNx, 120,
			XmNy, 10,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of label_LC_LET */
	label_LC_LET = XtVaCreateManagedWidget( "label_LC_LET",
			xmLabelWidgetClass,
			rowColumn1_LC,
			RES_CONVERT( XmNlabelString, "LET" ),
			NULL );


	/* Creation of textField_LC_VarName */
	textField_LC_VarName = XtVaCreateManagedWidget( "textField_LC_VarName",
			xmTextFieldWidgetClass,
			rowColumn1_LC,
			NULL );


	/* Creation of label_LC_Equals */
	label_LC_Equals = XtVaCreateManagedWidget( "label_LC_Equals",
			xmLabelWidgetClass,
			rowColumn1_LC,
			RES_CONVERT( XmNlabelString, "= (" ),
			NULL );


	/* Creation of textField_LC_1 */
	textField_LC_1 = XtVaCreateManagedWidget( "textField_LC_1",
			xmTextFieldWidgetClass,
			rowColumn1_LC,
			XmNcolumns, 5,
			NULL );


	/* Creation of label_LC_Times1 */
	label_LC_Times1 = XtVaCreateManagedWidget( "label_LC_Times1",
			xmLabelWidgetClass,
			rowColumn1_LC,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField_LC_2 */
	textField_LC_2 = XtVaCreateManagedWidget( "textField_LC_2",
			xmTextFieldWidgetClass,
			rowColumn1_LC,
			XmNcolumns, 10,
			NULL );


	/* Creation of labe5_LC */
	labe5_LC = XtVaCreateManagedWidget( "labe5_LC",
			xmLabelWidgetClass,
			rowColumn1_LC,
			XmNx, 470,
			XmNy, 20,
			XmNwidth, 10,
			XmNheight, 10,
			RES_CONVERT( XmNlabelString, ")" ),
			NULL );


	/* Creation of rowColumn2_LC */
	rowColumn2_LC = XtVaCreateWidget( "rowColumn2_LC",
			xmRowColumnWidgetClass,
			rowColumn_LC,
			XmNwidth, 20,
			XmNheight, 20,
			XmNx, 580,
			XmNy, 10,
			XmNorientation, XmHORIZONTAL,
			NULL );


	/* Creation of textField_LC_3 */
	textField_LC_3 = XtVaCreateManagedWidget( "textField_LC_3",
			xmTextFieldWidgetClass,
			rowColumn2_LC,
			XmNcolumns, 5,
			NULL );


	/* Creation of label_LC_Times2 */
	label_LC_Times2 = XtVaCreateManagedWidget( "label_LC_Times2",
			xmLabelWidgetClass,
			rowColumn2_LC,
			RES_CONVERT( XmNlabelString, "*" ),
			NULL );


	/* Creation of textField_LC_4 */
	textField_LC_4 = XtVaCreateManagedWidget( "textField_LC_4",
			xmTextFieldWidgetClass,
			rowColumn2_LC,
			XmNcolumns, 10,
			NULL );


	/* Creation of label6_LC */
	label6_LC = XtVaCreateManagedWidget( "label6_LC",
			xmLabelWidgetClass,
			rowColumn2_LC,
			XmNx, 270,
			XmNy, 10,
			XmNwidth, 10,
			XmNheight, 10,
			RES_CONVERT( XmNlabelString, "(" ),
			NULL );


	/* Creation of label7_LC */
	label7_LC = XtVaCreateManagedWidget( "label7_LC",
			xmLabelWidgetClass,
			rowColumn2_LC,
			XmNx, 290,
			XmNy, 10,
			XmNwidth, 10,
			XmNheight, 10,
			RES_CONVERT( XmNlabelString, ")" ),
			NULL );


	/* Creation of rowColumn_Select2_LC */
	rowColumn_Select2_LC = XtVaCreateWidget( "rowColumn_Select2_LC",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_LC,
			XmNleftOffset, 514,
			NULL );


	/* Creation of rowColumn_Select1_LC */
	rowColumn_Select1_LC = XtVaCreateWidget( "rowColumn_Select1_LC",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_LC,
			XmNleftOffset, 274,
			NULL );


	/* Creation of rowColumn_Select1_EXP */
	rowColumn_Select1_EXP = XtVaCreateWidget( "rowColumn_Select1_EXP",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_EXP,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNleftOffset, 274,
			NULL );


	/* Creation of rowColumn_Select1_F */
	rowColumn_Select1_F = XtVaCreateWidget( "rowColumn_Select1_F",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_F,
			XmNleftOffset, 381,
			NULL );


	/* Creation of rowColumn_Select_var2 */
	rowColumn_Select_var2 = XtVaCreateWidget( "rowColumn_Select_var2",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_F,
			XmNleftOffset, 546,
			NULL );


	/* Creation of rowColumn_Select_var3 */
	rowColumn_Select_var3 = XtVaCreateWidget( "rowColumn_Select_var3",
			xmRowColumnWidgetClass,
			form1,
			XmNwidth, 70,
			XmNheight, 30,
			XmNleftAttachment, XmATTACH_POSITION,
			XmNtopAttachment, XmATTACH_WIDGET,
			XmNtopWidget, rowColumn_F,
			NULL );


	/* Creation of label_dset_var1 */
	label_dset_var1 = XtVaCreateWidget( "label_dset_var1",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_F,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select1_F,
			NULL );


	/* Creation of label_dset_var2 */
	label_dset_var2 = XtVaCreateWidget( "label_dset_var2",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_F,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select_var2,
			NULL );


	/* Creation of label_dset_var3 */
	label_dset_var3 = XtVaCreateWidget( "label_dset_var3",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_F,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select_var3,
			NULL );


	/* Creation of label_dset1_LC */
	label_dset1_LC = XtVaCreateWidget( "label_dset1_LC",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_LC,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select1_LC,
			NULL );


	/* Creation of label_dset2_LC */
	label_dset2_LC = XtVaCreateWidget( "label_dset2_LC",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_LC,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select2_LC,
			NULL );


	/* Creation of label_dset1_EXP */
	label_dset1_EXP = XtVaCreateWidget( "label_dset1_EXP",
			xmLabelWidgetClass,
			form1,
			XmNbottomAttachment, XmATTACH_WIDGET,
			XmNbottomWidget, rowColumn_EXP,
			XmNleftAttachment, XmATTACH_OPPOSITE_WIDGET,
			XmNleftWidget, rowColumn_Select1_EXP,
			NULL );

	XtVaSetValues(label_LeftParen_1var,
			XmNpositionIndex, 0,
			NULL );

	XtVaSetValues(textField1_1var,
			XmNpositionIndex, 1,
			NULL );

	XtVaSetValues(label_Times_1var,
			XmNpositionIndex, 2,
			NULL );

	XtVaSetValues(label1_var2,
			XmNpositionIndex, 0,
			NULL );

	XtVaSetValues(textField1_var2,
			XmNpositionIndex, 1,
			NULL );

	XtVaSetValues(label2_var1,
			XmNpositionIndex, 2,
			NULL );

	XtVaSetValues(label1_var3,
			XmNpositionIndex, 0,
			NULL );

	XtVaSetValues(textField1_var3,
			XmNpositionIndex, 1,
			NULL );

	XtVaSetValues(label2_var3,
			XmNpositionIndex, 2,
			NULL );

	XtVaSetValues(rowColumn1_LC,
			XmNpositionIndex, 0,
			NULL );

	XtVaSetValues(label6_LC,
			XmNpositionIndex, 0,
			NULL );



	return ( JC_DefineVariable );
}

/*******************************************************************************
       The following is the 'Interface function' which is the
       external entry point for creating this interface.
       This function should be called from your application or from
       a callback function.
*******************************************************************************/

Widget	create_JC_DefineVariable( swidget _UxUxParent, int _Uxfunctional_form )
{
	Widget                  rtrn;

	UxParent = _UxUxParent;
	functional_form = _Uxfunctional_form;

	rtrn = _Uxbuild_JC_DefineVariable();

	JC_DV_Initialize((int)functional_form);
	XtPopup(UxGetWidget(rtrn), no_grab);
	
	return(rtrn);
}

/*******************************************************************************
       END OF FILE
*******************************************************************************/

