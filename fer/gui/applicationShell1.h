
/*******************************************************************************
       applicationShell1.h
       This header file is included by applicationShell1.c

*******************************************************************************/

#ifndef	_APPLICATIONSHELL1_INCLUDED
#define	_APPLICATIONSHELL1_INCLUDED


#include <stdio.h>
#include "UxLib.h"
#include "UxLabel.h"
#include "UxBboard.h"
#include "UxFrame.h"
#include "UxForm.h"
#include "UxApplSh.h"

/*******************************************************************************
       The definition of the context structure:
       If you create multiple copies of your interface, the context
       structure ensures that your callbacks use the variables for the
       correct copy.

       For each swidget in the interface, each argument to the Interface
       function, and each variable in the Interface Specific section of the
       Declarations Editor, there is an entry in the context structure.
       and a #define.  The #define makes the variable name refer to the
       corresponding entry in the context structure.
*******************************************************************************/

typedef	struct
{
	swidget	UxapplicationShell1;
	swidget	Uxform1;
	swidget	Uxframe1;
	swidget	UxbulletinBoard1;
	swidget	Uxlabel1;
	swidget	Uxlabel2;
	swidget	Uxlabel3;
	swidget	Uxframe2;
	swidget	Uxform2;
	swidget	Uxlabel7;
	swidget	Uxlabel8;
	swidget	Uxlabel9;
	swidget	Uxframe3;
	swidget	Uxform3;
	swidget	Uxlabel4;
	swidget	Uxlabel5;
	swidget	Uxlabel6;
	swidget	UxUxParent;
} _UxCapplicationShell1;

#ifdef CONTEXT_MACRO_ACCESS
static _UxCapplicationShell1   *UxApplicationShell1Context;
#define applicationShell1       UxApplicationShell1Context->UxapplicationShell1
#define form1                   UxApplicationShell1Context->Uxform1
#define frame1                  UxApplicationShell1Context->Uxframe1
#define bulletinBoard1          UxApplicationShell1Context->UxbulletinBoard1
#define label1                  UxApplicationShell1Context->Uxlabel1
#define label2                  UxApplicationShell1Context->Uxlabel2
#define label3                  UxApplicationShell1Context->Uxlabel3
#define frame2                  UxApplicationShell1Context->Uxframe2
#define form2                   UxApplicationShell1Context->Uxform2
#define label7                  UxApplicationShell1Context->Uxlabel7
#define label8                  UxApplicationShell1Context->Uxlabel8
#define label9                  UxApplicationShell1Context->Uxlabel9
#define frame3                  UxApplicationShell1Context->Uxframe3
#define form3                   UxApplicationShell1Context->Uxform3
#define label4                  UxApplicationShell1Context->Uxlabel4
#define label5                  UxApplicationShell1Context->Uxlabel5
#define label6                  UxApplicationShell1Context->Uxlabel6
#define UxParent                UxApplicationShell1Context->UxUxParent

#endif /* CONTEXT_MACRO_ACCESS */


/*******************************************************************************
       Declarations of global functions.
*******************************************************************************/

swidget	create_applicationShell1( swidget _UxUxParent );

#endif	/* _APPLICATIONSHELL1_INCLUDED */
