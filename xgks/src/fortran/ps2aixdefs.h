/*
 *		Copyright IBM Corporation 1989
 *
 *                      All Rights Reserved
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose and without fee is hereby granted,
 * provided that the above copyright notice appear in all copies and that
 * both that copyright notice and this permission notice appear in
 * supporting documentation, and that the name of IBM not be
 * used in advertising or publicity pertaining to distribution of the
 * software without specific, written prior permission.
 *
 * IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
 * ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
 * IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
 * ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
 * ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOFTWARE.
 *
 * FORTRAN to C binding for XGKS
 *
 * Bruce Greer
 * TCS Development
 * Cambridge MA
 *
 * June 19 1989
 *
 * $Id$
 * $__Header$
 */

#ifndef	AIXDEFS_H_INCLUDED
#define	AIXDEFS_H_INCLUDED

#define gopks_		gopks
#define gclks_		gclks
#define gopwk_		gopwk
#define gclwk_		gclwk
#define gacwk_		gacwk
#define gdawk_		gdawk
#define gclrwk_		gclrwk
#define grsgwk_		grsgwk
#define guwk_		guwk
#define gsds_		gsds
#define gmsg_		gmsg
#define gmsgs_		gmsgs
#define geclks_		geclks
#define gerhnd_		gerhnd
#define gerlog_		gerlog
#define gescid_		gescid
#define gesfv_		gesfv
#define gessbs_		gessbs
#define gesscm_		gesscm
#define gessdc_		gessdc
#define gesspn_		gesspn
#define gessrp_		gessrp
#define gessrn_		gessrn
#define gesver_		gesver
#define gwait_		gwait
#define gflush_		gflush
#define ggtlc_		ggtlc
#define ggtsk_		ggtsk
#define grqvl_		grqvl
#define ggtvl_		ggtvl
#define ggtch_		ggtch
#define ggtpk_		ggtpk
#define ggtst_		ggtst
#define ggtsts_		ggtsts
#define ginlc_		ginlc
#define ginsk_		ginsk
#define ginvl_		ginvl
#define ginch_		ginch
#define ginpk_		ginpk
#define ginst_		ginst
#define ginsts_		ginsts
#define gslcm_		gslcm
#define gsskm_		gsskm
#define gsvlm_		gsvlm
#define gschm_		gschm
#define gspkm_		gspkm
#define gsstm_		gsstm
#define grqlc_		grqlc
#define grqsk_		grqsk
#define grqvl_		grqvl
#define grqch_		grqch
#define grqpk_		grqpk
#define grqst_		grqst
#define grqsts_		grqsts
#define gsmlc_		gsmlc
#define gsmsk_		gsmsk
#define gsmvl_		gsmvl
#define gsmch_		gsmch
#define gsmpk_		gsmpk
#define gsmst_		gsmst
#define gsmsts_		gsmsts
#define gqiqov_		gqiqov
#define gqlvks_		gqlvks
#define gqewk_		gqewk
#define gqwkm_		gqwkm
#define gqmntn_		gqmntn
#define gqopwk_		gqopwk
#define gqacwk_		gqacwk
#define gqpli_		gqpli
#define gqpmi_		gqpmi
#define gqtxi_		gqtxi
#define gqchh_		gqchh
#define gqchup_		gqchup
#define gqchw_		gqchw
#define gqchb_		gqchb
#define gqtxp_		gqtxp
#define gqtxal_		gqtxal
#define gqfai_		gqfai
#define gqpa_		gqpa
#define gqparf_		gqparf
#define gqpkid_		gqpkid
#define gqln_		gqln
#define gqlwsc_		gqlwsc
#define gqplci_		gqplci
#define gqmk_		gqmk
#define gqmksc_		gqmksc
#define gqpmci_		gqpmci
#define gqtxfp_		gqtxfp
#define gqchxp_		gqchxp
#define gqchsp_		gqchsp
#define gqtxci_		gqtxci
#define gqfais_		gqfais
#define gqfasi_		gqfasi
#define gqfaci_		gqfaci
#define gqasf_		gqasf
#define gqcntn_		gqcntn
#define gqentn_		gqentn
#define gqnt_		gqnt
#define gqclip_		gqclip
#define gqopsg_		gqopsg
#define gqsgus_		gqsgus
#define gqsim_		gqsim
#define gqpxad_		gqpxad
#define gqpxa_		gqpxa
#define gqpx_		gqpx
#define gqsga_		gqsga
#define gqaswk_		gqaswk
#define gqops_		gqops
#define gqwkca_		gqwkca
#define gqwkcl_		gqwkcl
#define gqdsp_		gqdsp
#define gqdwka_		gqdwka
#define gqplf_		gqplf
#define gqpmf_		gqpmf
#define gqfaf_		gqfaf
#define gqpaf_		gqpaf
#define gqcf_		gqcf
#define gqppmr_		gqppmr
#define gqpplr_		gqpplr
#define gqtxf_		gqtxf
#define gqpcr_		gqpcr
#define gqlwk_		gqlwk
#define gqsgp_		gqsgp
#define gqdsga_		gqdsga
#define gqli_		gqli
#define gqptxr_		gqptxr
#define gqpfar_		gqpfar
#define gqppar_		gqppar
#define gqdlc_		gqdlc
#define gqdsk_		gqdsk
#define gqdvl_		gqdvl
#define gqdch_		gqdch
#define gqdpk_		gqdpk
#define gqdst_		gqdst
#define gqdds_		gqdds
#define gqegdp_		gqegdp
#define gqgdp_		gqgdp
#define gqwkc_		gqwkc
#define gqwks_		gqwks
#define gqwkdu_		gqwkdu
#define gqepli_		gqepli
#define gqplr_		gqplr
#define gqepmi_		gqepmi
#define gqpmr_		gqpmr
#define gqetxi_		gqetxi
#define gqtxr_		gqtxr
#define gqtxx_		gqtxx
#define gqtxxs_		gqtxxs
#define gqefai_		gqefai
#define gqfar_		gqfar
#define gqepai_		gqepai
#define gqpar_		gqpar
#define gqeci_		gqeci
#define gqcr_		gqcr
#define gqwkt_		gqwkt
#define gqsgwk_		gqsgwk
#define gqlcs_		gqlcs
#define gqsks_		gqsks
#define gqchs_		gqchs
#define gqpks_		gqpks
#define gqsts_		gqsts
#define gqvls_		gqvls
#define gqstss_		gqstss
#define gwitm_		gwitm
#define ggtitm_		ggtitm
#define grditm_		grditm
#define giitm_		giitm
#define gpl_		gpl
#define gpm_		gpm
#define gtx_		gtx
#define gtxs_		gtxs
#define gfa_		gfa
#define gca_		gca
#define ggdp_		ggdp
#define gsplr_		gsplr
#define gspmr_		gspmr
#define gstxr_		gstxr
#define gsfar_		gsfar
#define gspar_		gspar
#define gscr_		gscr
#define gssgt_		gssgt
#define gsvis_		gsvis
#define gshlit_		gshlit
#define gssgp_		gssgp
#define gsdtec_		gsdtec
#define gcrsg_		gcrsg
#define gclsg_		gclsg
#define grensg_		grensg
#define gdsg_		gdsg
#define gdsgwk_		gdsgwk
#define gasgwk_		gasgwk
#define gcsgwk_		gcsgwk
#define ginsg_		ginsg
#define gswn_		gswn
#define gsvp_		gsvp
#define gsvpip_		gsvpip
#define gselnt_		gselnt
#define gsclip_		gsclip
#define gswkwn_		gswkwn
#define gswkvp_		gswkvp
#define gprec_		gprec
#define gurec_		gurec
#define gevtm_		gevtm
#define gactm_		gactm
#define gspli_		gspli
#define gsln_		gsln
#define gslwsc_		gslwsc
#define gsplci_		gsplci
#define gspmi_		gspmi
#define gsmk_		gsmk
#define gsmksc_		gsmksc
#define gspmci_		gspmci
#define gstxi_		gstxi
#define gstxfp_		gstxfp
#define gschxp_		gschxp
#define gschsp_		gschsp
#define gstxci_		gstxci
#define gschh_		gschh
#define gschup_		gschup
#define gstxp_		gstxp
#define gstxal_		gstxal
#define gsfai_		gsfai
#define gsfais_		gsfais
#define gsfasi_		gsfasi
#define gsfaci_		gsfaci
#define gsasf_		gsasf
#define gspkid_		gspkid
#define gspa_		gspa
#define gsparf_		gsparf
#define gxconfig_	gxconfig
#define gxname_		gxname

#define inqlun_		inqlun

#endif	/* AIXDEFS_H_INCLUDED not defined */
