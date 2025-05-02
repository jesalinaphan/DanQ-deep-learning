#!/bin/bash
'''
 Downloads the mm10 genome and the following peaks from ENCODE:
 CTCF, POLR2A, GATA1, EP300, TAL1, MAFK, MAX, USF1, CHD2, E2F4, HCFC1,
 POLR2AphosphoS2, POLR2AphosphoS5, RAD21, TCF12, USF2, ZC3H11A, ZNF384, BHLHE40,
 CHD1, ELF1, ETS1, GABPA, JUND, KAT2A, MAZ, MEF2A, MXI1, MYC, NELFE,
 NRF1, RCOR1, SIN3A SMC3 TBP UBTF ZKSCAN1 ZMIZ1 MYB
'''

#NOTE: should not be necessary if running on oscar but i had to brew install wget
mkdir -p ./data/peaks

wget "https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz"
gunzip mm10.fa.gz
mv mm10.fa ./data

# Final file path for genome: mm10.fa

wget "https://www.encodeproject.org/files/ENCFF223ASW/@@download/ENCFF223ASW.bed.gz" 
gunzip ENCFF223ASW.bed.gz
mv ENCFF223ASW.bed ./data/peaks/CTCF.bed

wget "https://www.encodeproject.org/files/ENCFF228IWF/@@download/ENCFF228IWF.bed.gz"
gunzip ENCFF228IWF.bed.gz
mv ENCFF228IWF.bed ./data/peaks/POLR2A.bed

wget "https://www.encodeproject.org/files/ENCFF841DLH/@@download/ENCFF841DLH.bed.gz"
gunzip ENCFF841DLH.bed.gz
mv ENCFF841DLH.bed ./data/peaks/GATA1.bed

wget "https://www.encodeproject.org/files/ENCFF941FDR/@@download/ENCFF941FDR.bed.gz"
gunzip ENCFF941FDR.bed.gz
mv ENCFF941FDR.bed ./data/peaks/EP300.bed

wget "https://www.encodeproject.org/files/ENCFF893PQE/@@download/ENCFF893PQE.bed.gz"
gunzip ENCFF893PQE.bed.gz
mv ENCFF893PQE.bed ./data/peaks/TAL1.bed

wget "https://www.encodeproject.org/files/ENCFF031DDU/@@download/ENCFF031DDU.bed.gz"
gunzip ENCFF031DDU.bed.gz
mv ENCFF031DDU.bed ./data/peaks/MAFK.bed

wget "https://www.encodeproject.org/files/ENCFF262ITC/@@download/ENCFF262ITC.bed.gz"
gunzip ENCFF262ITC.bed.gz
mv ENCFF262ITC.bed ./data/peaks/MAX.bed

wget "https://www.encodeproject.org/files/ENCFF281FXQ/@@download/ENCFF281FXQ.bed.gz"
gunzip ENCFF281FXQ.bed.gz
mv ENCFF281FXQ.bed ./data/peaks/USF1.bed

wget "https://www.encodeproject.org/files/ENCFF399UKJ/@@download/ENCFF399UKJ.bed.gz"
gunzip ENCFF399UKJ.bed.gz
mv ENCFF399UKJ.bed ./data/peaks/CHD2.bed

wget "https://www.encodeproject.org/files/ENCFF734ZCR/@@download/ENCFF734ZCR.bed.gz"
gunzip ENCFF734ZCR.bed.gz
mv ENCFF734ZCR.bed ./data/peaks/E2F4.bed

wget "https://www.encodeproject.org/files/ENCFF625UEM/@@download/ENCFF625UEM.bed.gz"
gunzip ENCFF625UEM.bed.gz
mv ENCFF625UEM.bed ./data/peaks/HCFC1.bed

wget "https://www.encodeproject.org/files/ENCFF950CJS/@@download/ENCFF950CJS.bed.gz"
gunzip ENCFF950CJS.bed.gz
mv ENCFF950CJS.bed ./data/peaks/POLR2AphosphoS2.bed

wget "https://www.encodeproject.org/files/ENCFF527VCY/@@download/ENCFF527VCY.bed.gz"
gunzip ENCFF527VCY.bed.gz
mv ENCFF527VCY.bed ./data/peaks/POLR2AphosphoS5.bed

wget "https://www.encodeproject.org/files/ENCFF791EKG/@@download/ENCFF791EKG.bed.gz"
gunzip ENCFF791EKG.bed.gz
mv ENCFF791EKG.bed ./data/peaks/RAD21.bed

wget "https://www.encodeproject.org/files/ENCFF196FFI/@@download/ENCFF196FFI.bed.gz"
gunzip ENCFF196FFI.bed.gz
mv ENCFF196FFI.bed ./data/peaks/TCF12.bed

wget "https://www.encodeproject.org/files/ENCFF968IBZ/@@download/ENCFF968IBZ.bed.gz"
gunzip ENCFF968IBZ.bed.gz
mv ENCFF968IBZ.bed ./data/peaks/USF2.bed

wget "https://www.encodeproject.org/files/ENCFF235YHK/@@download/ENCFF235YHK.bed.gz"
gunzip ENCFF235YHK.bed.gz
mv ENCFF235YHK.bed  ./data/peaks/ZC3H11A.bed

wget "https://www.encodeproject.org/files/ENCFF611SZM/@@download/ENCFF611SZM.bed.gz"
gunzip ENCFF611SZM.bed.gz
mv ENCFF611SZM.bed ./data/peaks/ZNF384.bed

wget "https://www.encodeproject.org/files/ENCFF981ENW/@@download/ENCFF981ENW.bed.gz"
gunzip ENCFF981ENW.bed.gz
mv ENCFF981ENW.bed ./data/peaks/BHLHE40.bed

wget "https://www.encodeproject.org/files/ENCFF254IPE/@@download/ENCFF254IPE.bed.gz"
gunzip ENCFF254IPE.bed.gz
mv ENCFF254IPE.bed ./data/peaks/CHD1.bed

wget "https://www.encodeproject.org/files/ENCFF453OVN/@@download/ENCFF453OVN.bed.gz"
gunzip ENCFF453OVN.bed.gz
mv ENCFF453OVN.bed ./data/peaks/ELF1.bed

wget "https://www.encodeproject.org/files/ENCFF510QZL/@@download/ENCFF510QZL.bed.gz"
gunzip ENCFF510QZL.bed.gz
mv ENCFF510QZL.bed ./data/peaks/ETS1.bed

wget "https://www.encodeproject.org/files/ENCFF627TGO/@@download/ENCFF627TGO.bed.gz"
gunzip ENCFF627TGO.bed.gz
mv ENCFF627TGO.bed ./data/peaks/GABPA.bed

wget "https://www.encodeproject.org/files/ENCFF589AAH/@@download/ENCFF589AAH.bed.gz"
gunzip ENCFF589AAH.bed.gz
mv ENCFF589AAH.bed ./data/peaks/JUND.bed

wget "https://www.encodeproject.org/files/ENCFF672FKP/@@download/ENCFF672FKP.bed.gz"
gunzip ENCFF672FKP.bed.gz 
mv ENCFF672FKP.bed ./data/peaks/KAT2A.bed

wget "https://www.encodeproject.org/files/ENCFF980MPW/@@download/ENCFF980MPW.bed.gz"
gunzip ENCFF980MPW.bed.gz
mv ENCFF980MPW.bed ./data/peaks/MAZ.bed

wget "https://www.encodeproject.org/files/ENCFF044OUI/@@download/ENCFF044OUI.bed.gz"
gunzip ENCFF044OUI.bed.gz
mv ENCFF044OUI.bed ./data/peaks/MEF2A.bed

wget "https://www.encodeproject.org/files/ENCFF067LMI/@@download/ENCFF067LMI.bed.gz"
gunzip ENCFF067LMI.bed.gz
mv ENCFF067LMI.bed ./data/peaks/MXI1.bed

wget "https://www.encodeproject.org/files/ENCFF573INC/@@download/ENCFF573INC.bed.gz"
gunzip ENCFF573INC.bed.gz
mv ENCFF573INC.bed ./data/peaks/MYC.bed

wget "https://www.encodeproject.org/files/ENCFF416TBW/@@download/ENCFF416TBW.bed.gz"
gunzip ENCFF416TBW.bed.gz
mv ENCFF416TBW.bed ./data/peaks/NELFE.bed

wget "https://www.encodeproject.org/files/ENCFF443MGH/@@download/ENCFF443MGH.bed.gz"
gunzip ENCFF443MGH.bed.gz
mv ENCFF443MGH.bed ./data/peaks/NRF1.bed

wget "https://www.encodeproject.org/files/ENCFF567VSU/@@download/ENCFF567VSU.bed.gz"
gunzip ENCFF567VSU.bed.gz
mv ENCFF567VSU.bed ./data/peaks/RCOR1.bed

wget "https://www.encodeproject.org/files/ENCFF468PLO/@@download/ENCFF468PLO.bed.gz"
gunzip ENCFF468PLO.bed.gz
mv ENCFF468PLO.bed ./data/peaks/SIN3A.bed

wget "https://www.encodeproject.org/files/ENCFF525MNL/@@download/ENCFF525MNL.bed.gz"
gunzip ENCFF525MNL.bed.gz
mv ENCFF525MNL.bed ./data/peaks/SMC3.bed

wget "https://www.encodeproject.org/files/ENCFF700PEG/@@download/ENCFF700PEG.bed.gz"
gunzip ENCFF700PEG.bed.gz
mv ENCFF700PEG.bed ./data/peaks/TBP.bed

wget "https://www.encodeproject.org/files/ENCFF813PFO/@@download/ENCFF813PFO.bed.gz"
gunzip ENCFF813PFO.bed.gz
mv ENCFF813PFO.bed ./data/peaks/UBTF.bed

wget "https://www.encodeproject.org/files/ENCFF066CEV/@@download/ENCFF066CEV.bed.gz"
gunzip ENCFF066CEV.bed.gz
mv ENCFF066CEV.bed ./data/peaks/ZKSCAN1.bed

wget "https://www.encodeproject.org/files/ENCFF839UAT/@@download/ENCFF839UAT.bed.gz"
gunzip ENCFF839UAT.bed.gz
mv ENCFF839UAT.bed ./data/peaks/ZMIZ1.bed

wget "https://www.encodeproject.org/files/ENCFF911NHJ/@@download/ENCFF911NHJ.bed.gz"
gunzip ENCFF911NHJ.bed.gz
mv ENCFF911NHJ.bed ./data/peaks/MYB.bed