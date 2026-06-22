#inspiration by nanoGPT configuration way
#it is simple and elegant
root_data_path='/root/autodl-tmp/evosnr_0605/data/'
save_dir='/root/autodl-tmp/evosnr_0605/Weights'
# #sino config
# target='Sinorhizobium meliloti 1021'
# type_filter='Chr'

# #Ecol config
# target='Escherichia coli str K-12 substr. MG1655'
# type_filter='chr'

# #kleb config
# target='Klebsiella aerogenes KCTC 2190'
# type_filter='chr'

# #Shig config
# target='Shigella flexneri 5a str. M90T'
# type_filter='chr'

#Agro config
# target='Agrobacterium tumefaciens str C58'
# type_filter='circular-Chr'


#source='Sinorhizobium meliloti 1021'
source='Escherichia coli str K-12 substr. MG1655'
target='Agrobacterium tumefaciens str C58'



Lexicon_source_path=root_data_path+source[:4]+f'/motifs/lexicon.txt'
Lexicon_target_path=root_data_path+target[:4]+f'/motifs/lexicon.txt'

FastTextModel_source_path=root_data_path+source[:4]+f'/fasttext_model/model.bin'
FastTextModel_target_path=root_data_path+target[:4]+f'/fasttext_model/model.bin'

PCA_split_path=root_data_path+target[:4]+f'/500_split/'
Source_PCA_path=root_data_path+source[:4]+f'/500_split/split_train.csv'


