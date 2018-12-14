import glob
import FWCore.ParameterSet.Config as cms


process = cms.Process("run")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# Get file names with unix pattern expander
#files = glob.glob("/afs/cern.ch/work/b/bregnery/public/HHwwwwMCgenerator/CMSSW_8_0_21/src/hhMCgenerator/RootFiles/M3500/*.root")
# Add to the beginning of each filename
#for ifile in range(len(files)):
#    files[ifile] = "file:" + files[ifile]

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
	#'root://cms-xrd-global.cern.ch//store/mc/RunIISpring16MiniAODv2/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/70000/847868B8-7E26-E611-9BEA-02163E01413E.root'
	#'/store/mc/RunIISpring16MiniAODv2/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/70000/847868B8-7E26-E611-9BEA-02163E01413E.root'
#	'/store/mc/RunIISpring16MiniAODv2/QCD_Pt_600to800_TuneCUETP8M1_13TeV_pythia8/MINIAODSIM/PUSpring16_80X_mcRun2_asymptotic_2016_miniAODv2_v0-v1/70000/065ACA52-3D1C-E611-97A4-44A84225CFF0.root'
	'/store/mc/RunIISummer16MiniAODv2/QCD_Pt_1400to1800_TuneCUETP8M1_13TeV_pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6-v1/110000/E0C50E83-BFB1-E611-9228-0025905B85D2.root '
	)
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.selectedAK8Jets = cms.EDFilter('PATJetSelector',
    src = cms.InputTag('slimmedJetsAK8'),
    cut = cms.string('pt > 100.0 && abs(eta) < 2.4'),
    filter = cms.bool(True)
)

process.countAK8Jets = cms.EDFilter("PATCandViewCountFilter",
    minNumber = cms.uint32(1),
    maxNumber = cms.uint32(99999),
    src = cms.InputTag("selectedAK8Jets"),
    filter = cms.bool(True)
)

process.run = cms.EDProducer('HHESTIAProducer',
	inputJetColl = cms.string('selectedAK8Jets'),
        isSignal = cms.bool(False)
)

process.TFileService = cms.Service("TFileService", fileName = cms.string("preprocess_HHESTIA_QCD.root") )

process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string("ana_out.root"),
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               outputCommands = cms.untracked.vstring('drop *',
                                                                      'keep *_*run*_*_*'
                                                                      #, 'keep *_goodPatJetsCATopTagPF_*_*'
                                                                      #, 'keep recoPFJets_*_*_*'
                                                                      ) 
                               )
process.outpath = cms.EndPath(process.out)

process.p = cms.Path(process.selectedAK8Jets*process.countAK8Jets*process.run)
