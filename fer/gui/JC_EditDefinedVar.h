
/*******************************************************************************
       JC_EditDefinedVar.h
       This header file is included by JC_EditDefinedVar.c

*******************************************************************************/

#ifndef	_JC_EDITDEFINEDVAR_INCLUDED
#define	_JC_EDITDEFINEDVAR_INCLUDED


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

extern Widget	JC_EditDefinedVar;
extern Widget	EDV_rowColumn_Select;
extern Widget	EDV_textField_definition;

/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

Widget	create_JC_EditDefinedVar( swidget _UxUxParent );

#endif	/* _JC_EDITDEFINEDVAR_INCLUDED */
