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



static void DisableThickBtn()
{
  XtUnmanageChild(optionMenu5);
  XtManageChild(optionMenu18);
}

static void EnableThickBtn()
{
  XtManageChild(optionMenu5);
  XtUnmanageChild(optionMenu18);
}


static void InitArrays()
{

	styleWidgets[0] = UxGetWidget(lineStyle2_b8);
	styleWidgets[1] = UxGetWidget(lineStyle1_b9);
	styleWidgets[2] = UxGetWidget(lineStyle1_b10);
	styleWidgets[3] = UxGetWidget(lineStyle1_b11);
	styleWidgets[4] = UxGetWidget(lineStyle1_b12);
	styleWidgets[5] = UxGetWidget(lineStyle1_b13);
	styleWidgets[6] = UxGetWidget(lineStyle1_b14); 

	thickWidgets[0] = UxGetWidget(lineStyle2_b1);
	thickWidgets[1] = UxGetWidget(lineStyle1_b1);
	thickWidgets[2] = UxGetWidget(lineStyle1_b2);

	symbolWidgets[0] = UxGetWidget(lineStyle_b1);
	symbolWidgets[1] = UxGetWidget(optionMenu_p_b10);
	symbolWidgets[2] = UxGetWidget(lineStyle_b89);
	symbolWidgets[3] = UxGetWidget(lineStyle_b91);
	symbolWidgets[4] = UxGetWidget(lineStyle_b92);
	symbolWidgets[5] = UxGetWidget(lineStyle_b93);
	symbolWidgets[6] = UxGetWidget(lineStyle_b94);
	symbolWidgets[7] = UxGetWidget(lineStyle_b95);
	symbolWidgets[8] = UxGetWidget(lineStyle_b96);
	symbolWidgets[9] = UxGetWidget(lineStyle_b97);
	symbolWidgets[10] = UxGetWidget(lineStyle_b98);
	symbolWidgets[11] = UxGetWidget(lineStyle_b99);
	symbolWidgets[12] = UxGetWidget(lineStyle_b100);
	symbolWidgets[13] = UxGetWidget(lineStyle_b101);
	symbolWidgets[14] = UxGetWidget(lineStyle_b102);
	symbolWidgets[15] = UxGetWidget(lineStyle_b103);
	symbolWidgets[16] = UxGetWidget(lineStyle_b104);
	symbolWidgets[17] = UxGetWidget(lineStyle_b105);
	symbolWidgets[18] = UxGetWidget(lineStyle_b106);
	symbolWidgets[19] = UxGetWidget(lineStyle_b107);
	symbolWidgets[20] = UxGetWidget(lineStyle_b108);
	symbolWidgets[21] = UxGetWidget(lineStyle_b109);
	symbolWidgets[22] = UxGetWidget(lineStyle_b110);
	symbolWidgets[23] = UxGetWidget(lineStyle_b111);
	symbolWidgets[24] = UxGetWidget(lineStyle_b112);
	symbolWidgets[25] = UxGetWidget(lineStyle_b113);
	symbolWidgets[26] = UxGetWidget(lineStyle_b114);
	symbolWidgets[27] = UxGetWidget(lineStyle_b115);
	symbolWidgets[28] = UxGetWidget(lineStyle_b116);
	symbolWidgets[29] = UxGetWidget(lineStyle_b117);
	symbolWidgets[30] = UxGetWidget(lineStyle_b118);
	symbolWidgets[31] = UxGetWidget(lineStyle_b119);
	symbolWidgets[32] = UxGetWidget(lineStyle_b120);
	symbolWidgets[33] = UxGetWidget(lineStyle_b121);
	symbolWidgets[34] = UxGetWidget(lineStyle_b122);
	symbolWidgets[35] = UxGetWidget(lineStyle_b123);
	symbolWidgets[36] = UxGetWidget(lineStyle_b124);
	symbolWidgets[37] = UxGetWidget(lineStyle_b125);
	symbolWidgets[38] = UxGetWidget(lineStyle_b126);
	symbolWidgets[39] = UxGetWidget(lineStyle_b127);
	symbolWidgets[40] = UxGetWidget(lineStyle_b128);
	symbolWidgets[41] = UxGetWidget(lineStyle_b129);
	symbolWidgets[42] = UxGetWidget(lineStyle_b130);
	symbolWidgets[43] = UxGetWidget(lineStyle_b131);
	symbolWidgets[44] = UxGetWidget(lineStyle_b132);
	symbolWidgets[45] = UxGetWidget(lineStyle_b133);
	symbolWidgets[46] = UxGetWidget(lineStyle_b134);
	symbolWidgets[47] = UxGetWidget(lineStyle_b135);
	symbolWidgets[48] = UxGetWidget(lineStyle_b136);
	symbolWidgets[49] = UxGetWidget(lineStyle_b137);
	symbolWidgets[50] = UxGetWidget(lineStyle_b138);
	symbolWidgets[51] = UxGetWidget(lineStyle_b139);
	symbolWidgets[52] = UxGetWidget(lineStyle_b140);
	symbolWidgets[53] = UxGetWidget(lineStyle_b141);
	symbolWidgets[54] = UxGetWidget(lineStyle_b142);
	symbolWidgets[55] = UxGetWidget(lineStyle_b143);
	symbolWidgets[56] = UxGetWidget(lineStyle_b144);
	symbolWidgets[57] = UxGetWidget(lineStyle_b145);
	symbolWidgets[58] = UxGetWidget(lineStyle_b146);
	symbolWidgets[59] = UxGetWidget(lineStyle_b147);
	symbolWidgets[60] = UxGetWidget(lineStyle_b148);
	symbolWidgets[61] = UxGetWidget(lineStyle_b149);
	symbolWidgets[62] = UxGetWidget(lineStyle_b150);
	symbolWidgets[63] = UxGetWidget(lineStyle_b151);
	symbolWidgets[64] = UxGetWidget(lineStyle_b152);
	symbolWidgets[65] = UxGetWidget(lineStyle_b153);
	symbolWidgets[66] = UxGetWidget(lineStyle_b154);
	symbolWidgets[67] = UxGetWidget(lineStyle_b155);
	symbolWidgets[68] = UxGetWidget(lineStyle_b156);
	symbolWidgets[69] = UxGetWidget(lineStyle_b157);
	symbolWidgets[70] = UxGetWidget(lineStyle_b158);
	symbolWidgets[71] = UxGetWidget(lineStyle_b159);
	symbolWidgets[72] = UxGetWidget(lineStyle_b160);
	symbolWidgets[73] = UxGetWidget(lineStyle_b161);
	symbolWidgets[74] = UxGetWidget(lineStyle_b162);
	symbolWidgets[75] = UxGetWidget(lineStyle_b163);
	symbolWidgets[76] = UxGetWidget(lineStyle_b164);
	symbolWidgets[77] = UxGetWidget(lineStyle_b165);
	symbolWidgets[78] = UxGetWidget(lineStyle_b166);
	symbolWidgets[79] = UxGetWidget(lineStyle_b167);
	symbolWidgets[80] = UxGetWidget(lineStyle_b168);
	symbolWidgets[81] = UxGetWidget(lineStyle_b169);
	symbolWidgets[82] = UxGetWidget(lineStyle_b170);
	symbolWidgets[83] = UxGetWidget(lineStyle_b171);
	symbolWidgets[84] = UxGetWidget(lineStyle_b172);
	symbolWidgets[85] = UxGetWidget(lineStyle_b173);
	symbolWidgets[86] = UxGetWidget(lineStyle_b174);
	symbolWidgets[87] = UxGetWidget(lineStyle_b175);
	symbolWidgets[88] = UxGetWidget(lineStyle_b176);
}

void SetInitialPOState()
{
	XmToggleButtonSetState(UxGetWidget(toggleButton84), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton86), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton89), False, False);

	switch (gPlotTypeSelected) {
		case 1:
		case 2:
			UnmapPlotOptions();
			Map1DOptions();
			XmToggleButtonSetState(UxGetWidget(toggleButton84), True, False);
			localPlotType = 1;
			break;
		case 3:
		case 4:
		case 5:
			UnmapPlotOptions();
			Map2DOptions();
			XmToggleButtonSetState(UxGetWidget(toggleButton86), True, False);
			localPlotType = 3;
			break;
		case 6:
			UnmapPlotOptions();
			MapVectorOptions();
			XmToggleButtonSetState(UxGetWidget(toggleButton89), True, False);
			localPlotType = 6;
			break;
			
	}
	PlotOptions2Interface();
}

static void InitPixmaps()
{
	XtVaSetValues(UxGetWidget(styleWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_1_xpm),
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_1_ins_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_2_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[3]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_3_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[4]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_4_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[5]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_5_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(styleWidgets[6]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(lines_6_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[0]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_single_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_single_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_double_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_double_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(thickWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelInsensitivePixmap, GetPixmapFromData(lines_triple_ins_xpm),
		XmNlabelPixmap, GetPixmapFromData(lines_triple_xpm),
		NULL);

#ifdef FULL_GUI_VERSION
	XtVaSetValues(UxGetWidget(symbolWidgets[1]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_1_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[2]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_2_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[3]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_3_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[4]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_4_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[5]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_5_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[6]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_6_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[7]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_7_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[8]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_8_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[9]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_9_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[10]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_10_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[11]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_11_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[12]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_12_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[13]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_13_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[14]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_14_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[15]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_15_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[16]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_16_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[17]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_17_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[18]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_18_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[19]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_19_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[20]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_20_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[21]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_21_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[22]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_22_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[23]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_23_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[24]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_24_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[25]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_25_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[26]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_26_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[27]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_27_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[28]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_28_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[29]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_29_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[30]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_30_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[31]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_31_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[32]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_32_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[33]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_33_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[34]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_34_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[35]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_35_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[36]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_36_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[37]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_37_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[38]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_38_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[39]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_39_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[40]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_40_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[41]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_41_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[42]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_42_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[43]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_43_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[44]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_44_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[45]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_45_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[46]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_46_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[47]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_47_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[48]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_48_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[49]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_49_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[50]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_50_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[51]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_51_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[52]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_52_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[53]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_53_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[54]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_54_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[55]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_55_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[56]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_56_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[57]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_57_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[58]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_58_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[59]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_59_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[60]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_60_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[61]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_61_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[62]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_62_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[63]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_63_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[64]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_64_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[65]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_65_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[66]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_66_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[67]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_67_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[68]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_68_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[69]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_69_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[70]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_70_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[71]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_71_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[72]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_72_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[73]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_73_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[74]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_74_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[75]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_75_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[76]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_76_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[77]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_77_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[78]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_78_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[79]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_79_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[80]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_80_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[81]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_81_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[82]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_82_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[83]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_83_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[84]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_84_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[85]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_85_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[86]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_86_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[87]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_87_xpm),
		NULL);

	XtVaSetValues(UxGetWidget(symbolWidgets[88]),
		XmNlabelType, XmPIXMAP,
		XmNlabelPixmap, GetPixmapFromData(sym_88_xpm),
		NULL);
#endif
}

UpdateLineStyleCB(wid, client_data, cbs)
Widget wid;
XtPointer client_data;
XtPointer cbs;
{
	char *tempText;
	XmString buttonLabel;
	int val, oldStyle;
	
	/* option is encoded in button label */
	XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
	XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);

	if (tempText[0] == 'A') {
		/* auto selected--turn off the line thickness menu */
		oneDPlotOptions.style = 0;
		DisableThickBtn();
	}
	else {
	        EnableThickBtn();
		sscanf(tempText, "%d", &val);
		oldStyle = oneDPlotOptions.style;

		/* isolate just the line style */
		if (oneDPlotOptions.style > 6 && oneDPlotOptions.style < 13)
			oneDPlotOptions.style -= 6;
		else if (oneDPlotOptions.style >= 13)
			oneDPlotOptions.style -= 12;
		
		if ((val == 0 || val == 7 || val == 13) && oneDPlotOptions.style > 0)
			/* selection from the line thickness menu */
			oneDPlotOptions.style += val-1;
		else {
			/* selection from style menu--restore line thickness too (if any) */
			if (oldStyle > 6 && oldStyle < 13)
				/* double line */
				oneDPlotOptions.style = val + 6;
			else if (oldStyle >= 13)
				/* triple line */
				oneDPlotOptions.style = val + 12;
			else
				oneDPlotOptions.style = val;
		}
	}
	XtFree(tempText);
}

UpdateLineSymbolCB(wid, client_data, cbs)
Widget wid;
XtPointer client_data;
XtPointer cbs;
{
	char *tempText;
	XmString buttonLabel;
	int val;

	/* option is encoded in button label */
	XtVaGetValues(wid,
		XmNlabelString, &buttonLabel,
		NULL);
	XmStringGetLtoR(buttonLabel, XmSTRING_DEFAULT_CHARSET, &tempText);

	if (tempText[0] == 'A')
		oneDPlotOptions.symbol = 0;
	else {
		sscanf(tempText, "%d", &val);
		oneDPlotOptions.symbol = val;
	}
	XtFree(tempText);
}


Update1DOptionsCB()
{
	if (XmToggleButtonGetState(UxGetWidget(toggleButton83)))
		oneDPlotOptions.autoMode = 1;
	else
		oneDPlotOptions.autoMode = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton47)))
		/* transpose */
		oneDPlotOptions.transpose = 1;
	else 
		oneDPlotOptions.transpose = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton46)))
		/* plot labels */
		oneDPlotOptions.labels = 1;
	else 
		oneDPlotOptions.labels = 0;
}

Update2DOptionsCB()
{
	Boolean isSet;
	char *tText;
	float val;

	tText = (char *)malloc(32);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton79))) {
		/* autoscale levels */
		twoDPlotOptions.levels = 0;
                XtVaSetValues(UxGetWidget(textField40), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField46), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField47), 
			XmNvalue, "",
			NULL);
	}
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton80))) {
		/* Reuse last */
		twoDPlotOptions.levels = 1;
                XtVaSetValues(UxGetWidget(textField40), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField46), 
			XmNvalue, "",
			NULL);
                 XtVaSetValues(UxGetWidget(textField47), 
			XmNvalue, "",
			NULL);
	}
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton81))) {
		/* custom levels */
		twoDPlotOptions.levels = 2;

		/* get the value of low */
		tText = XmTextFieldGetString(UxGetWidget(textField40));
		if (strlen(tText) == 0)
			customContourOptions.low = -99;
		else {
			sscanf(tText, "%f", &val);
			customContourOptions.low = val;
		}

		/* get the value of high */
		tText = XmTextFieldGetString(UxGetWidget(textField46));
		if (strlen(tText) == 0)
			customContourOptions.high = -99;
		else {
			sscanf(tText, "%f", &val);
			customContourOptions.high = val;
		}

		/* get the value of delta */
		tText = XmTextFieldGetString(UxGetWidget(textField47));
		if (strlen(tText) == 0)
			customContourOptions.delta = -99;
		else {
			sscanf(tText, "%f", &val);
			customContourOptions.delta = val;
		}
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton82)))
		/* use a color key */
		twoDPlotOptions.colorKey = 1;
	else 
		twoDPlotOptions.colorKey = 0;


	if (XmToggleButtonGetState(UxGetWidget(toggleButton78)) &&
		XtIsSensitive(UxGetWidget(toggleButton78)))
		/* overlay contour lines */
		twoDPlotOptions.overlayContours = 1;
	else 
		twoDPlotOptions.overlayContours = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton47)))
		/* transpose */
		twoDPlotOptions.transpose = 1;
	else 
		twoDPlotOptions.transpose = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton46)))
		/* plot labels */
		twoDPlotOptions.labels = 1;
	else 
		twoDPlotOptions.labels = 0;
	XtFree(tText);
}

UpdateVectorOptionsCB()
{
	char *tText;
	int val;

	tText = (char *)malloc(32);

	if (XmToggleButtonGetState(UxGetWidget(toggleButton48)))
		/* use aspect correction */
		vectorOptions.aspectCorrection = 1;
	else 
		vectorOptions.aspectCorrection = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton66)))
		/* autoscale levels */
		vectorOptions.vectorLength = 1;
	else if (XmToggleButtonGetState(UxGetWidget(toggleButton67)))
		/* Reuse last length */
		vectorOptions.vectorLength = 2;
	else
		/* custom length */
		 vectorOptions.vectorLength = 3;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton54)))
		/* auto x skip */
		vectorOptions.customXSkip = 0;
	else
		/* custom x skip */
		vectorOptions.customXSkip = 1;

	/* get the value of y skip */
	tText = XmTextFieldGetString(UxGetWidget(textField15));
	if (strlen(tText) == 0)
	        vectorOptions.xSkip = 1;
	else {
		sscanf(tText, "%d", &val);
		vectorOptions.xSkip = val;
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton58)))
		/* auto y skip */
		vectorOptions.customYSkip = 0;
	else 
		/* custom y skip */
		vectorOptions.customYSkip = 1;

	/* get the value of y skip */
	tText = XmTextFieldGetString(UxGetWidget(textField16));
	if (strlen(tText) == 0)
	        vectorOptions.ySkip = 1;
	else {
		sscanf(tText, "%d", &val);
		vectorOptions.ySkip = val;
	}

	if (XmToggleButtonGetState(UxGetWidget(toggleButton47)))
		/* transpose */
		vectorOptions.transpose = 1;
	else 
		vectorOptions.transpose = 0;

	if (XmToggleButtonGetState(UxGetWidget(toggleButton46)))
		/* plot labels */
		vectorOptions.labels = 1;
	else 
		vectorOptions.labels = 0;
	XtFree(tText);
}

/* functions to show/hide plot options frames */
 UnmapPlotOptions()
{
	XtUnmapWidget(UxGetWidget(frame13));
	XtUnmapWidget(UxGetWidget(frame15));
	XtUnmapWidget(UxGetWidget(frame18));
}

 Map1DOptions()
{
	XtMapWidget(UxGetWidget(frame13));
}

 Map2DOptions()
{
	XtMapWidget(UxGetWidget(frame18));
}

 MapVectorOptions()
{
	XtMapWidget(UxGetWidget(frame15));
}

 MapVectorStickOptions()
{
	;
}

void PlotOptions2Interface()
{
	Widget wid;
	char *tText;

	tText = (char *)malloc(32);

	/* Make interface reflect state of plot options */

	/* 1D */
	/* set the line style option menu */
	if (dOneDPlotOptions.style <= 6)
		wid = styleWidgets[dOneDPlotOptions.style];
	else if (dOneDPlotOptions.style <= 12)
		wid = styleWidgets[dOneDPlotOptions.style-6];
	else if (dOneDPlotOptions.style <= 18)
		wid = styleWidgets[dOneDPlotOptions.style-12];

	XtVaSetValues(UxGetWidget(optionMenu15),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu15));
	XtMapWidget(UxGetWidget(optionMenu15));

	/* line thickness option menu */
	if (dOneDPlotOptions.style <= 6)
		wid = thickWidgets[0];
	else if (dOneDPlotOptions.style <= 12)
		wid = thickWidgets[1];
	else if (dOneDPlotOptions.style <= 18)
		wid = thickWidgets[2];

	XtVaSetValues(UxGetWidget(optionMenu5),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu5));
	XtMapWidget(UxGetWidget(optionMenu5));

	if (dOneDPlotOptions.style)
	        DisableThickBtn();
	else
	        EnableThickBtn();

	/* symbol option menu */
	wid = symbolWidgets[dOneDPlotOptions.symbol];
	XtVaSetValues(UxGetWidget(optionMenu19),
		XmNmenuHistory, wid,
		NULL);
	XtUnmapWidget(UxGetWidget(optionMenu19));
	XtMapWidget(UxGetWidget(optionMenu19));

	if (dOneDPlotOptions.autoMode) {
		XmToggleButtonSetState(UxGetWidget(toggleButton83), True, False);
		XtSetSensitive(UxGetWidget(optionMenu15), False);
		XtSetSensitive(UxGetWidget(optionMenu19), False);
	        DisableThickBtn();
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton83), False, False);
		XtSetSensitive(UxGetWidget(optionMenu15), True);
		XtSetSensitive(UxGetWidget(optionMenu19), True);
	        EnableThickBtn();
	}

	/* 2D */
	XmToggleButtonSetState(UxGetWidget(toggleButton79), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton80), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton81), False, False);
        XtVaSetValues(UxGetWidget(textField40), 
		XmNvalue, "",
		NULL);
       	XtVaSetValues(UxGetWidget(textField46), 
		XmNvalue, "",
		NULL);
	XtVaSetValues(UxGetWidget(textField47), 
		XmNvalue, "",
		NULL);

	if (dTwoDPlotOptions.levels == 0) {
		/* autoscale levels */
		XmToggleButtonSetState(UxGetWidget(toggleButton79), True, False);
		XtSetSensitive(UxGetWidget(textField40), False);
		XtSetSensitive(UxGetWidget(textField46), False);
		XtSetSensitive(UxGetWidget(textField47), False);
		XtSetSensitive(UxGetWidget(label70), False);
		XtSetSensitive(UxGetWidget(label71), False);
		XtSetSensitive(UxGetWidget(label78), False);
	}
	else if (dTwoDPlotOptions.levels == 1) {
		/* Reuse last */
		XmToggleButtonSetState(UxGetWidget(toggleButton80), True, False);
		XtSetSensitive(UxGetWidget(textField40), False);
		XtSetSensitive(UxGetWidget(textField46), False);
		XtSetSensitive(UxGetWidget(textField47), False);
		XtSetSensitive(UxGetWidget(label70), False);
		XtSetSensitive(UxGetWidget(label71), False);
		XtSetSensitive(UxGetWidget(label78), False);
	}
	else if (dTwoDPlotOptions.levels == 2) {
		/* custom levels */
		XtSetSensitive(UxGetWidget(textField40), True);
		XtSetSensitive(UxGetWidget(textField46), True);
		XtSetSensitive(UxGetWidget(textField47), True);
		XtSetSensitive(UxGetWidget(label70), True);
		XtSetSensitive(UxGetWidget(label71), True);
		XtSetSensitive(UxGetWidget(label78), True);

		XmToggleButtonSetState(UxGetWidget(toggleButton81), True, False);

		/* set the value of low */
		sprintf(tText, "%f", dCustomContourOptions.low);
		XmTextFieldSetString(UxGetWidget(textField40), tText);

		/* set the value of high */
		sprintf(tText, "%f", dCustomContourOptions.high);
		XmTextFieldSetString(UxGetWidget(textField46), tText);

		/* set the value of delta */
		sprintf(tText, "%f", dCustomContourOptions.delta);
		XmTextFieldSetString(UxGetWidget(textField47), tText);
	}

	if (dTwoDPlotOptions.colorKey)
		/* use a color key */
		XmToggleButtonSetState(UxGetWidget(toggleButton82), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton82), False, False);

	if (dTwoDPlotOptions.overlayContours)
		/* overlay contour lines */
		XmToggleButtonSetState(UxGetWidget(toggleButton78), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton78), False, False);

	/* Vector */	
	if (dVectorOptions.aspectCorrection)
		/* use aspect correction */
		XmToggleButtonSetState(UxGetWidget(toggleButton48), True, False);
	else 
		XmToggleButtonSetState(UxGetWidget(toggleButton48), False, False);

	XmToggleButtonSetState(UxGetWidget(toggleButton66), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton67), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton77), False, False);
	if (dVectorOptions.vectorLength == 1)
		/* autoscale levels */
		XmToggleButtonSetState(UxGetWidget(toggleButton66), True, False);
	else if (dVectorOptions.vectorLength == 2)
		/* Reuse last length */
		XmToggleButtonSetState(UxGetWidget(toggleButton67), True, False);
	else
		/* custom length */
		XmToggleButtonSetState(UxGetWidget(toggleButton77), True, False);


	XmToggleButtonSetState(UxGetWidget(toggleButton54), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton57), False, False);
	if (dVectorOptions.customXSkip == 0) {
		/* auto x skip */
		XmToggleButtonSetState(UxGetWidget(toggleButton54), True, False);
		XtUnmapWidget(UxGetWidget(textField15));
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton57), True, False);
		/* set the value of x skip */
		sprintf(tText, "%d", (int)dVectorOptions.xSkip);
		XmTextFieldSetString(UxGetWidget(textField15), tText);
		XtMapWidget(UxGetWidget(textField15));
	}

	XmToggleButtonSetState(UxGetWidget(toggleButton58), False, False);
	XmToggleButtonSetState(UxGetWidget(toggleButton65), False, False);
	if (dVectorOptions.customYSkip == 0)
		/* auto y skip */ {
		XmToggleButtonSetState(UxGetWidget(toggleButton58), True, False);
		XtUnmapWidget(UxGetWidget(textField16));
	}
	else {
		XmToggleButtonSetState(UxGetWidget(toggleButton65), True, False);
		/* set the value of y skip */
		sprintf(tText, "%d", (int)dVectorOptions.ySkip);
		XmTextFieldSetString(UxGetWidget(textField16), tText);
		XtMapWidget(UxGetWidget(textField16));
	}

	/* transpose and labels */
	if (localPlotType == 1) {
		if (dOneDPlotOptions.transpose)
			XmToggleButtonSetState(UxGetWidget(toggleButton47), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton47), False, False);

		if (dOneDPlotOptions.labels)
			XmToggleButtonSetState(UxGetWidget(toggleButton46), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton46), False, False);
	}
	else if (localPlotType == 3) {
		if (dTwoDPlotOptions.transpose)
			XmToggleButtonSetState(UxGetWidget(toggleButton47), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton47), False, False);

		if (dTwoDPlotOptions.labels)
			XmToggleButtonSetState(UxGetWidget(toggleButton46), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton46), False, False);
	}
	else if (localPlotType == 6) {
		if (dVectorOptions.transpose)
			XmToggleButtonSetState(UxGetWidget(toggleButton47), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton47), False, False);

		if (dVectorOptions.labels)
			XmToggleButtonSetState(UxGetWidget(toggleButton46), True, False);
		else
			XmToggleButtonSetState(UxGetWidget(toggleButton46), False, False);
	}

	XtFree(tText);
}
