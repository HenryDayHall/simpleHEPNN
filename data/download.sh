# http://opendata.cern.ch/record/22073 <- signal
curl http://opendata.cern.ch/record/22073/files/assets/cms/mc/RunIIFall15MiniAODv2/ggZH_HToBB_ZToNuNu_M125_13TeV_powheg_herwigpp/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/00000/1C203AB1-0D24-E611-A3DF-0002C94CD13C.root --output signal.root
# http://opendata.cern.ch/record/18382 <-- background 
curl http://opendata.cern.ch/record/18382/files/assets/cms/mc/RunIIFall15MiniAODv2/QCD_bEnriched_HT1500to2000_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PU25nsData2015v1_76X_mcRun2_asymptotic_v12-v1/80000/36AFA6CE-E635-E611-B86A-44A8423DE404.root --output background.root
