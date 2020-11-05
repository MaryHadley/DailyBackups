import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing #https://twiki.cern.ch/CMSPublic/SWGuideAboutPythonConfigFile#Passing_Command_Line_Arguments_T

process = cms.Process("analysis")

options = VarParsing.VarParsing("analysis")



#List of Options
options.register( "applyZmuonFilter",
    True,
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Shall we prefilter out some events and never bother giving them to the analyzer? Defaults to True. If you change to false, you will need to, for the moment, comment out the indicated lines in the ZmuonAnalyzer.cc. Will try to make a nice boolean to avoid this later. Will flag with a comment 'If applyZmuonFilter=False, comment out!'"
)

options.register("isMC",
    True, #set to false for crab test run
    VarParsing.VarParsing.multiplicity.singleton,
    VarParsing.VarParsing.varType.bool,
    "Is this simulation (aka MC) or is it data we have collected from the CMS detector? Defaults to True."
            
)

# This needs to go here, after we have defined all the options!
options.parseArguments()
print "Options are:", options

process.ZmuonAnalyzer = cms.EDAnalyzer("ZmuonAnalyzer",
   muonCollection = cms.InputTag("slimmedMuons"),
   electronCollection = cms.InputTag("slimmedElectrons"),
   bits = cms.InputTag("TriggerResults","", "HLT"),
   objects = cms.InputTag("selectedPatTrigger"),
   genParticles = cms.InputTag("prunedGenParticles"),
   vertexCollection = cms.InputTag("offlineSlimmedPrimaryVertices"),
   metTag = cms.InputTag("slimmedMETs"),
   pfCands = cms.InputTag("packedPFCandidates"),
   rho     = cms.InputTag("fixedGridRhoFastjetAll"), #Median energy density, see: https://arxiv.org/pdf/0707.1378.pdf
  #Might add Pileup here later if I think it is something we might change ever, but at the moment is hard-coded in the analyzer
   isMC   = cms.bool(options.isMC),
)

if options.applyZmuonFilter:
   process.ZmuonFilter = cms.EDFilter("ZmuonFilter",
     muonCollection = cms.InputTag("slimmedMuons"),
     bits = cms.InputTag("TriggerResults","", "HLT"),
     objects = cms.InputTag("selectedPatTrigger"),
#     genParticles = cms.InputTag("prunedGenParticles"), #not needed here
#     pfCands = cms.InputTag("packedPFCandidates")  #not needed here
     pTCut = cms.double(2.),
     etaCut = cms.double(3.),
     invMass4MuCut_low = cms.double(60.),
     invMass4MuCut_high = cms.double(120.),
  )
   process.nEventsTotal = cms.EDProducer("EventCountProducer")
   process.nEventsFiltered = cms.EDProducer("EventCountProducer")

#process.maxEvents.input = 1000

#process.maxEvents = cms.untracked.PSet(
#   input = cms.untracked.int32(40000)
#   input = cms.untracked.int32(-1)
 #  input = cms.untracked.int32(1000) #for crab test, just look at 1000 events

#)


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#                                          "file:../miniAOD_01.root",
#                                          "file:../miniAOD_02.root",
#                                          "file:../miniAOD_03.root",
#                                          "file:../miniAOD_04.root",
#                                          "file:../miniAOD_05.root",
#                                          "file:../miniAOD_06.root"
#                                          "file:../../../miniAOD_lowStats.root",
#                                           "file:../SingleMu_Run2018A-17Sep2018-v2_F8CDAAA9-11C7-A34F-A059-409CF95EB82A.root",
#                                           "file:../SingleMu_2017B-31Mar2018-v1_FC2B7874-F538-E811-9C29-0025905A60A8.root",
#                                            "file:0079D4A1-71DE-AF4B-90E4-115037F02923.root", #DiMu Run18 A
 #                                             "file:FD881AE8-B6AC-8E4F-9453-2D5E6135D476.root" #DiMu Run18 A
 #                                            "file:E36DDD08-CC3D-6D49-8C71-C6E8C201A2E5.root", #this is DiMU 2018RunB data
 #                                            "file:../YZ_4MuFS_DPS_APV_MiniAOD_2016UL_Yfirst_Zsecond_03_10_2020.root",
 #                                           "file:../YZ_4MuFS_DPS_APV_MiniAOD_2016UL_Zfirst_Ysecond_03_10_2020.root",
#                                           "file:../YZ_4MuFS_DPS_MiniAOD_2016UL_Yfirst_Zsecond_03_10_2020.root",
                                          "file:../YZ_4MuFS_DPS_MiniAOD_2016UL_Zfirst_Ysecond_03_10_2020.root",


                                    ),
   duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

process.maxEvents = cms.untracked.PSet(
#   input = cms.untracked.int32(40000)
  input = cms.untracked.int32(-1)
   # input = cms.untracked.int32(1000) #for crab test, just look at 1000 events

)

process.TFileService = cms.Service("TFileService",
   fileName = cms.string("mc_ZUpsi_YZ_4MuFS_DPS_MiniAOD_2016UL_Zfirst_Ysecond_05_11_2020_remove_save_event_req_for_MC.root")
)

#process.maxEvents.input = 1000
#process.maxEvents = cms.untracked.PSet(
#   input = cms.untracked.int32(40000)
#   input = cms.untracked.int32(-1)
#    input = cms.untracked.int32(1000) #for crab test, just look at 1000 events

#)

process.options = cms.untracked.PSet(
   wantSummary = cms.untracked.bool(True),
#   SkipEvent = cms.untracked.vstring('ProductNotFound')
)


process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

#process.MessageLogger = cms.Service("MessageLogger",
#                    destinations   = cms.untracked.vstring('messages.txt')
#)

if options.applyZmuonFilter:
    process.p = cms.Path(
         process.nEventsTotal *process.ZmuonFilter *process.nEventsFiltered * process.ZmuonAnalyzer 
        )

else:


    process.p = cms.Path(
        process.ZmuonAnalyzer
)

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data')

