
root_species_path='/root/autodl-tmp/zwk/evosnr_0605/preprocessing/Species/'
root_data_path='/root/autodl-tmp/zwk/evosnr_0605/data/'
cache_path='/root/autodl-tmp/zwk/evosnr_0605/preprocessing/cache/'


#sino config
# target='Sinorhizobium meliloti 1021'
# type_filter='Chr'

# #Ecol config
# target='Escherichia coli str K-12 substr. MG1655'
# type_filter='chr'

# #kleb config
# target='Klebsiella aerogenes KCTC 2190'
# type_filter='chr'

#Shig config
# target='Shigella flexneri 5a str. M90T'
# type_filter='chr'

#Agro config
target='Agrobacterium tumefaciens str C58'
type_filter='circular-Chr'


target_fna=root_species_path+target[:4]+f'/{target}'+'.fna'
target_prom=root_species_path+target[:4]+'/'+target+' promoters.csv'

target_train_PCA=root_data_path+target[:4]+f'/split/split_train.csv'
target_PCA_output=root_data_path+target[:4]+f'/all_PCA.csv'
fasttext_model_output=root_data_path+target[:4]+f'/fasttext_model/model.bin'



streme_output_dir=root_data_path+target[:4]+f'/motifs/streme/'
streme_path=root_data_path+target[:4]+f'/motifs/streme/streme.txt'
lexicon_path=root_data_path+target[:4]+f'/motifs/lexicon.txt'