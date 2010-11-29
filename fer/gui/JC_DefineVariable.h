
/*******************************************************************************
       JC_DefineVariable.h
       This header file is included by JC_DefineVariable.c

*******************************************************************************/

#ifndef	_JC_DEFINEVARIABLE_INCLUDED
#define	_JC_DEFINEVARIABLE_INCLUDED


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

extern Widget	JC_DefineVariable;
extern Widget	rowColumn_Select2_LC;
extern Widget	rowColumn_Select1_LC;
extern Widget	rowColumn_Select1_EXP;
extern Widget	rowColumn_Select1_F;
extern Widget	rowColumn_Select_var2;
extern Widget	rowColumn_Select_var3;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_JC_DefineVariable( swidget _UxUxParent, int _Uxfunctional_form );

#endif	/* _JC_DEFINEVARIABLE_INCLUDED */
