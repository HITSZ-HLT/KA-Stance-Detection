read -p "input training dataset: [p_stance, sem16, covid_19, vast]: " trainDataset
read -p "input train dataset mode: [single_target, zero_shot]: " trainData
read -p "input model name: [roberta_base, roberta_large, bertweet_base, bertweet_large, ct_bert_large]: " trainModel
read -p "input model framework: [base, kasd]: " framework
read -p "input running mode: [sweep, wandb, normal]: " runMode
read -p "input training cuda idx: " cudaIdx

trainName="bert_based"
train_Dir="baselines/bert_based"

currTime=$(date +"%Y-%m-%d_%T")
fileName="${train_Dir}/${trainName}_main.py"
outputDir="logs/${trainName}/${trainData}"

if [ ! -d ${outputDir} ]; then
    mkdir -p ${outputDir}
fi

outputName="${outputDir}/${trainDataset}_${trainModel}_${framework}_${currTime}.log"
nohup python ${fileName} --cuda_idx ${cudaIdx} --dataset_name ${trainDataset} --model_name ${trainModel} --${trainData} --framework_name ${framework} --${runMode} > ${outputName} 2>&1 &